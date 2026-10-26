/*
 * Frontend for the voice interaction backend.
 *
 * Captures microphone audio via the Web Audio API, resamples it to the
 * 16 kHz mono float32 format the backend's WebSocket protocol expects
 * (see backend/server.py's module docstring for the full protocol), and
 * streams it as binary frames as it is captured, rather than recording a
 * full clip and sending it once, since that streaming behavior is what the
 * whole real time requirement depends on.
 *
 * A deliberate simplification for this pass: audio playback of the
 * assistant's reply happens on the server machine's own speakers
 * (tts.speak_text() plays there directly), not through this page. This UI
 * shows captions live instead. Streaming synthesized audio back to the
 * browser is a reasonable next step, noted in README.md, but out of scope
 * here since the target environment for this demo is one machine.
 */

const SAMPLE_RATE = 16000;
const CHUNK_SECONDS = 0.25;
const CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS;
const CALIBRATION_SECONDS = 1.0; // matches backend/server.py's calibration window

// Translated text for the language chosen on the first screen. SltI18n falls
// back to English, then to the key, so this never returns an empty string.
function tr(key, vars) {
  try {
    return window.SltI18n ? window.SltI18n.t(key, vars) : key;
  } catch (err) {
    return key;
  }
}
function num(value) {
  return window.SltFormat ? window.SltFormat.integer(value) : String(value);
}

const botAvatar = document.getElementById("bot-avatar");
const transcript = document.getElementById("transcript");
const backendUrlInput = document.getElementById("backend-url");

let audioContext = null;
let mediaStream = null;
let sourceNode = null;
let processorNode = null;
let socket = null;
let resampleRemainder = new Float32Array(0);
let sendBuffer = new Float32Array(0);
let calibrationSamplesSent = 0;
let calibrationDone = false;
let isActive = false;
let isConnecting = false;
let isMuted = false;
// Gates whether captured audio actually gets sent over the WebSocket (see
// setStatus() below, which flips this per status event, and
// handleAudioProcess(), which checks it, alongside isMuted). Audio is
// only sent while the server has told us it is actually listening
// ("listening" or the initial "calibrating"), never while it is
// "thinking" or "speaking". On this project's single machine demo setup,
// the assistant's spoken reply plays on that same machine's speakers, so
// a microphone that kept streaming through "speaking" could physically
// pick up the assistant's own voice and have it transcribed and treated
// as the next thing the user said. This is what actually prevents that:
// nothing captured during those two states ever reaches the server at
// all, not just a server side filter on top of an already sent, already
// ambiguous chunk.
let audioSendAllowed = false;
// Barge in over /ws/voice: the server plays the reply on its own speakers, so the
// microphone is also sent while the status is "speaking" and the server decides
// (stricter VAD, see backend/vad.py BargeInDetector) whether the user really
// talks over it. Not before the first customer turn (the introduction and the
// calibration must stay clean) and not for the canned closing lines.
let bargeInReady = false;
let bargeInBlocked = false;
let currentPartialBubble = null;
let selectedLanguage = null;
let selectedMode = null; // "voice" | "chat", set when leaving the select screen
let userName = null; // null means Guest; set from the verified account for an existing user
// Session token from POST /auth/verify, kept in memory only (never in
// localStorage). The backend also sets an httpOnly cookie, which is what
// identifies the user when this page is served by the backend itself; the
// token covers the development setup where the page comes from another
// port and that cookie is not sent.
let sessionToken = null;
let greetingsGiven = { voice: false, chat: false }; // avoids re-greeting on every back-and-forth navigation
// The LiveKit room handle (see livekit-voice.js) while a conversation runs on
// that transport; null on the original /ws/voice path.
let livekitSession = null;
// The Gemini Live session (see live-voice.js) while a conversation runs on tier 1.
let liveSession = null;
// The last lines of the conversation, kept so a move to another voice tier can carry them.
const recentTurns = [];
function rememberTurn(who, text) {
  if (!text) return;
  recentTurns.push(`${who}: ${String(text).slice(0, 200)}`);
  while (recentTurns.length > 6) recentTurns.shift();
}
function handoverContext() {
  return recentTurns.join("\n").slice(-800);
}

// Male/female choice for the avatar AND the spoken voice, kept in the browser.
// The backend voice styles are "female_natural" and "male" (backend/tts.py).
function loadAvatarGender() {
  try {
    return localStorage.getItem("avatarGender") === "male" ? "male" : "female";
  } catch (err) {
    return "female";
  }
}
let avatarGender = loadAvatarGender();

function voiceStyle() {
  return avatarGender === "male" ? "male" : "female_natural";
}

// The new avatar renderer (frontend/avatar), when it is the active one.
// window.avatarBridge (avatar/loader.js) is safe to call at any time and is
// only "active" once the renderer really runs, so the legacy avatar is
// untouched whenever AVATAR_RENDERER is legacy or the new one fell back.
function newAvatar() {
  const bridge = window.avatarBridge;
  return bridge && bridge.isActive() ? bridge : null;
}

// Where the backend is. When this page is served by the backend itself
// (the single app, python -m backend), everything is on the page's own
// origin. When it is opened from the separate static development server
// on port 5500, the hidden #backend-url field still decides.
function servedByBackend() {
  return window.location.protocol.startsWith("http") && window.location.port !== "5500";
}

function backendHttpBase() {
  if (servedByBackend()) return window.location.origin;
  return backendUrlInput.value.trim().replace(/^ws/i, "http").replace(/\/ws\/voice.*$/i, "");
}

function backendWsUrl() {
  if (servedByBackend()) {
    const scheme = window.location.protocol === "https:" ? "wss" : "ws";
    return `${scheme}://${window.location.host}/ws/voice`;
  }
  return backendUrlInput.value.trim();
}

function authHeaders() {
  return sessionToken ? { Authorization: `Bearer ${sessionToken}` } : {};
}

// Screen router: a language picker, a start chat / start voice choice, a
// new or existing user check, then the actual chat or voice screen. Plain
// hash based routing (#language, #select, #user, #voice, #chat) rather
// than a library, consistent with the rest of this frontend's no
// framework approach. The language screen is always the entry point, per
// the required flow, regardless of what hash a reload happens to land on.
const SCREEN_IDS = ["language", "select", "user", "voice", "chat"];

// The screen being left fades out for a moment (CSS .screen.leaving) before
// the next one fades in, instead of an abrupt cut. Skipped when the visitor
// asks for reduced motion. navToken makes a second navigation that starts
// during the fade win, so rapid clicks cannot leave two screens showing.
const SCREEN_FADE_MS = 120;
let navToken = 0;

// Five star rating (frontend/ui/rating): asked once when a chat or voice
// session ends, and only if the visitor actually took part in it.
const sessionTurns = { chat: 0, voice: 0 };

function askForRating(channel) {
  const turns = sessionTurns[channel] || 0;
  sessionTurns[channel] = 0;
  if (!window.SLTRating || turns < 1) return;
  try {
    SLTRating.ask({ channel: channel, language: selectedLanguage || "en", turns: turns });
  } catch (err) {
    /* the rating is optional; never block leaving the session */
  }
}

function navigateTo(screenName) {
  if (!SCREEN_IDS.includes(screenName)) screenName = "language";
  window.location.hash = screenName;
  const current = document.querySelector(".screen.active");
  const target = document.getElementById(`screen-${screenName}`);
  const token = ++navToken;

  const show = () => {
    if (token !== navToken) return;
    SCREEN_IDS.forEach((id) => {
      const el = document.getElementById(`screen-${id}`);
      if (el) {
        el.classList.remove("leaving");
        el.classList.toggle("active", id === screenName);
      }
    });
    window.scrollTo({ top: 0, behavior: "smooth" });
    // Responsive layer for the voice screen (frontend/ui/dashboard); a no-op
    // when the UI layer did not load.
    if (screenName === "voice" && window.SltUi && window.SltUi.dashboard) {
      try { window.SltUi.dashboard.attach(document.getElementById("screen-voice")); } catch (err) { /* decoration only */ }
    }
  };

  const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!current || current === target || reduceMotion) {
    show();
    return;
  }
  current.classList.add("leaving");
  setTimeout(show, SCREEN_FADE_MS);
}

document.querySelectorAll(".choice-button[data-language]").forEach((button) => {
  button.addEventListener("click", () => {
    selectedLanguage = button.getAttribute("data-language");
    // Sets <html lang>, the page title and every data-i18n text for this language.
    if (window.SltI18n) {
      window.SltI18n.setLanguage(selectedLanguage).then(repaintDynamicLabels).catch(() => {});
    }
    // Remembered so the landing greeting uses it on the next visit.
    try { localStorage.setItem("slt-language", selectedLanguage); } catch (err) { /* storage may be blocked */ }
    navigateTo("select");
  });
});

document.querySelectorAll("[data-nav]").forEach((el) => {
  el.addEventListener("click", (event) => {
    event.preventDefault();
    navigateTo(el.getAttribute("data-nav"));
  });
});

// New or existing user screen: gates both the chat and voice screens
// (reached from either "Start Chat" or "Start Voice") so the agent can
// address the visitor by name or as Guest. New user continues as Guest
// straight away. Existing user is verified: the landline number and PIN
// are checked by POST /auth/verify against the customer database, and the
// name the assistant uses comes from that account, resolved again by the
// backend from the session on every request rather than trusted from here.
const goChatButton = document.getElementById("go-chat");
const goVoiceButton = document.getElementById("go-voice");
const goExistingButton = document.getElementById("go-existing");
const goNewButton = document.getElementById("go-new");
const userChoiceButtons = document.getElementById("user-choice-buttons");
const existingNamePrompt = document.getElementById("existing-name-prompt");
const userPhoneInput = document.getElementById("user-phone-input");
const userPinInput = document.getElementById("user-pin-input");
const verifyError = document.getElementById("verify-error");
const confirmNameButton = document.getElementById("confirm-name");
const cancelExistingButton = document.getElementById("cancel-existing");

function showVerifyError(message) {
  if (!verifyError) return;
  verifyError.textContent = message || "";
  verifyError.classList.toggle("hidden", !message);
}

// One time code login (the alternative to the PIN; GET /auth/methods says
// whether the server offers it). otpMode is "pin" (the original form),
// "otp-phone" (asking for the number) or "otp-code" (a code was sent).
const useOtpButton = document.getElementById("use-otp");
const usePinButton = document.getElementById("use-pin");
const otpStep = document.getElementById("otp-step");
const otpInfo = document.getElementById("otp-info");
const otpInput = document.getElementById("user-otp-input");
const otpResendButton = document.getElementById("otp-resend");
const pinLabel = document.getElementById("user-pin-label");
let otpMode = "pin";
let otpChallengeId = null;
let otpOffered = false;
let otpResendTimer = null;

function setOtpMode(mode) {
  otpMode = mode;
  const pinVisible = mode === "pin";
  [pinLabel, userPinInput].forEach((el) => el && el.classList.toggle("hidden", !pinVisible));
  if (otpStep) otpStep.classList.toggle("hidden", mode !== "otp-code");
  if (userPhoneInput) userPhoneInput.disabled = mode === "otp-code";
  if (confirmNameButton) {
    confirmNameButton.textContent = tr(mode === "pin" ? "legacy.continue" : mode === "otp-phone" ? "legacy.send_code" : "legacy.verify");
  }
  if (useOtpButton) useOtpButton.classList.toggle("hidden", !(otpOffered && mode === "pin"));
  if (usePinButton) usePinButton.classList.toggle("hidden", !(otpOffered && mode !== "pin"));
  if (mode !== "otp-code") {
    clearInterval(otpResendTimer);
    otpChallengeId = null;
    if (otpInput) otpInput.value = "";
  }
  showVerifyError("");
}

async function loadAuthMethods() {
  try {
    const response = await fetch(`${backendHttpBase()}/auth/methods`);
    if (!response.ok) return;
    const data = await response.json();
    otpOffered = Array.isArray(data.methods) && data.methods.includes("otp") && data.otp_available === true;
  } catch (err) {
    otpOffered = false;
  }
  setOtpMode(otpMode);
}

function startResendCountdown(seconds) {
  clearInterval(otpResendTimer);
  let left = Math.max(0, seconds | 0);
  const paint = () => {
    if (!otpResendButton) return;
    otpResendButton.disabled = left > 0;
    otpResendButton.textContent = left > 0 ? tr("legacy.resend_in", { seconds: num(left) }) : tr("legacy.resend");
  };
  paint();
  otpResendTimer = setInterval(() => {
    left -= 1;
    paint();
    if (left <= 0) clearInterval(otpResendTimer);
  }, 1000);
}

// Plain words for each way the one time code steps can fail
// (see backend/otp/routes.py).
async function otpErrorMessage(response) {
  let data = {};
  try {
    data = await response.json();
  } catch (err) {
    /* no JSON body */
  }
  const wait = parseInt(response.headers.get("Retry-After") || data.retry_after || "0", 10);
  if (response.status === 422) {
    return tr("verify.errors.phone_format");
  }
  if (response.status === 423) return tr("legacy.err.code_locked");
  if (response.status === 429) {
    return wait > 0 ? tr("legacy.err.wait_seconds", { seconds: num(wait) }) : tr("legacy.err.too_many");
  }
  if (response.status === 503 || response.status === 404) {
    return tr("legacy.err.otp_unavailable");
  }
  return tr("legacy.err.code_wrong");
}

async function requestOtp() {
  const phone = userPhoneInput.value.trim();
  if (!phone) {
    showVerifyError(tr("legacy.err.enter_phone"));
    return;
  }
  confirmNameButton.disabled = true;
  showVerifyError("");
  try {
    const response = await fetch(`${backendHttpBase()}/auth/otp/request`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ phone, language: selectedLanguage }),
    });
    if (!response.ok) {
      showVerifyError(await otpErrorMessage(response));
      return;
    }
    const data = await response.json();
    setOtpMode("otp-code");
    otpChallengeId = data.challenge_id;
    if (otpInfo) {
      otpInfo.textContent = tr("legacy.otp_sent", { masked: data.masked_destination, minutes: num(Math.round(data.expires_in / 60)) });
    }
    startResendCountdown(data.resend_in);
    if (otpInput) otpInput.focus();
  } catch (err) {
    showVerifyError(tr("verify.errors.network"));
  } finally {
    confirmNameButton.disabled = false;
  }
}

async function verifyOtp() {
  const code = otpInput.value.trim();
  if (!/^[0-9]{4,8}$/.test(code)) {
    showVerifyError(tr("legacy.err.enter_code"));
    return;
  }
  confirmNameButton.disabled = true;
  showVerifyError("");
  try {
    const response = await fetch(`${backendHttpBase()}/auth/otp/verify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ challenge_id: otpChallengeId, code }),
    });
    if (!response.ok) {
      const message = await otpErrorMessage(response);
      if (response.status === 423) setOtpMode("otp-phone");
      showVerifyError(message);
      return;
    }
    const data = await response.json();
    sessionToken = data.token;
    userName = data.display_name;
    setOtpMode("pin");
    proceedToSelectedMode();
  } catch (err) {
    showVerifyError(tr("verify.errors.network"));
  } finally {
    confirmNameButton.disabled = false;
  }
}

function resetUserScreen() {
  if (userChoiceButtons) userChoiceButtons.classList.remove("hidden");
  if (existingNamePrompt) existingNamePrompt.classList.add("hidden");
  if (userPinInput) userPinInput.value = "";
  setOtpMode("pin");
  showVerifyError("");
}

// Plain words for each way verification can fail (see backend/auth/routes.py).
async function verifyErrorMessage(response) {
  let detail = "";
  try {
    const data = await response.json();
    detail = typeof data.detail === "string" ? data.detail : "";
  } catch (err) {
    /* no JSON body */
  }
  const retryAfter = parseInt(response.headers.get("Retry-After") || "0", 10);
  const minutes = Math.max(1, Math.ceil(retryAfter / 60));
  if (response.status === 423) {
    return tr("legacy.err.account_locked", { minutes: num(minutes) });
  }
  if (response.status === 429) {
    return tr("legacy.err.device_limit");
  }
  if (response.status === 422) {
    return tr("legacy.err.phone_pin_format");
  }
  return tr("legacy.err.mismatch");
}

async function verifyExistingUser() {
  if (otpMode === "otp-phone") return requestOtp();
  if (otpMode === "otp-code") return verifyOtp();
  const phone = userPhoneInput.value.trim();
  const pin = userPinInput.value.trim();
  if (!phone || !pin) {
    showVerifyError(tr("legacy.err.enter_phone_pin"));
    return;
  }
  confirmNameButton.disabled = true;
  showVerifyError("");
  try {
    const response = await fetch(`${backendHttpBase()}/auth/verify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ phone, pin }),
    });
    if (!response.ok) {
      showVerifyError(await verifyErrorMessage(response));
      return;
    }
    const data = await response.json();
    sessionToken = data.token;
    userName = data.display_name;
    userPinInput.value = "";
    proceedToSelectedMode();
  } catch (err) {
    showVerifyError(tr("verify.errors.network"));
  } finally {
    confirmNameButton.disabled = false;
  }
}

function goToUserScreen(mode) {
  selectedMode = mode;
  resetUserScreen();
  navigateTo("user");
}

if (goChatButton) goChatButton.addEventListener("click", () => goToUserScreen("chat"));
if (goVoiceButton) goVoiceButton.addEventListener("click", () => goToUserScreen("voice"));

if (goExistingButton) {
  goExistingButton.addEventListener("click", () => {
    // The three step verification wizard (frontend/verify). It sets the
    // user_session cookie itself; the server decides what that unlocks. If
    // the script did not load, the older PIN form below is used instead.
    if (window.SltVerify) {
      window.SltVerify.open({
        language: selectedLanguage,
        backendBase: backendHttpBase(),
        onVerified: (user) => {
          userName = user.display_name;
          proceedToSelectedMode();
        },
        onCancel: () => {},
      });
      return;
    }
    userChoiceButtons.classList.add("hidden");
    existingNamePrompt.classList.remove("hidden");
    showVerifyError("");
    userPhoneInput.focus();
  });
}

if (cancelExistingButton) {
  cancelExistingButton.addEventListener("click", resetUserScreen);
}

if (useOtpButton) useOtpButton.addEventListener("click", () => setOtpMode("otp-phone"));
if (usePinButton) usePinButton.addEventListener("click", () => setOtpMode("pin"));
if (otpResendButton) otpResendButton.addEventListener("click", requestOtp);
if (otpInput) {
  otpInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      confirmNameButton.click();
    }
  });
}
loadAuthMethods();

if (goNewButton) {
  goNewButton.addEventListener("click", () => {
    userName = null;
    sessionToken = null;
    proceedToSelectedMode();
  });
}

if (confirmNameButton) {
  confirmNameButton.addEventListener("click", verifyExistingUser);
}

[userPhoneInput, userPinInput].forEach((input) => {
  if (!input) return;
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      confirmNameButton.click();
    }
  });
});

// A canned greeting rather than one generated by the LLM: a fixed
// introduction is instant and reliable. This copy is only for the text
// chat screen, in the language picked on the first screen; on the voice
// screen the backend speaks its own copy of the same introductions out
// loud (see _GREETINGS in backend/server.py). "Guest" is used in the
// selected language when no name was given.
function greetingText() {
  return tr("chat.greeting", { name: userName || tr("chat.guest") });
}

function chatGreeting() {
  if (greetingsGiven.chat) return;
  greetingsGiven.chat = true;
  addBubble("assistant", greetingText(), tr("chat.intro_meta"), chatTranscript);
}

function proceedToSelectedMode() {
  const mode = selectedMode || "voice";
  navigateTo(mode);
  if (mode === "voice") {
    // The voice screen's own introduction (and the conversation itself)
    // starts from showMicPermissionModal()/beginVoiceSession() below, not
    // here: mic permission is asked first, and only after that is granted
    // does the assistant introduce itself and start listening. See those
    // functions for the full sequencing and why.
    showMicPermissionModal();
  } else {
    chatGreeting();
  }
}

// Always start on the language screen; a saved hash from a previous visit
// is not honored on a fresh load, matching the required first landing page.
navigateTo("language");

// Theme toggle (light/dark), persisted so it survives a reload. The
// initial theme is also applied by a small inline script in index.html's
// <head>, before this file loads, so there is no flash of the wrong theme.
const themeToggle = document.getElementById("theme-toggle");
const themeToggleIcon = document.getElementById("theme-toggle-icon");

function currentTheme() {
  return document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
}

// Single stroke line icons (matching every other icon in this app), swapped
// by replacing innerHTML rather than textContent, since these are markup,
// not emoji characters.
const ICON_SUN =
  '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" ' +
  'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="5"></circle>' +
  '<line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line>' +
  '<line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>' +
  '<line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line>' +
  '<line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>';
const ICON_MOON =
  '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" ' +
  'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
  '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>';

function applyThemeIcon() {
  if (themeToggleIcon) themeToggleIcon.innerHTML = currentTheme() === "dark" ? ICON_SUN : ICON_MOON;
}

if (themeToggle) {
  applyThemeIcon();
  themeToggle.addEventListener("click", () => {
    const next = currentTheme() === "dark" ? "light" : "dark";
    // Fade the colors for a moment (style.css .theme-fade) rather than flip.
    const root = document.documentElement;
    root.classList.add("theme-fade");
    setTimeout(() => root.classList.remove("theme-fade"), 320);
    root.setAttribute("data-theme", next);
    try {
      localStorage.setItem("theme", next);
    } catch (err) {
      /* localStorage unavailable; theme choice just will not persist */
    }
    applyThemeIcon();
  });
}

// Cursor sensitive background glow: writes the pointer position as CSS
// custom properties that style.css's .bg-glow radial gradient reads.
// Throttled to one update per animation frame so it stays smooth without
// flooding style recalculation on every mousemove event.
const bgGlow = document.querySelector(".bg-glow");
const GLOW_HALF_SIZE = 480; // half of .bg-glow's 960px box (style.css)
let glowUpdateQueued = false;
let lastPointer = { x: window.innerWidth / 2, y: window.innerHeight * 0.3 };

function applyGlowPosition() {
  glowUpdateQueued = false;
  // Moving the box with transform stays on the compositor; the old way,
  // writing CSS variables on the root, restyled the whole page per frame.
  if (bgGlow) {
    bgGlow.style.transform = `translate3d(${lastPointer.x - GLOW_HALF_SIZE}px, ${lastPointer.y - GLOW_HALF_SIZE}px, 0)`;
  }
}

window.addEventListener("mousemove", (event) => {
  lastPointer = { x: event.clientX, y: event.clientY };
  if (!glowUpdateQueued) {
    glowUpdateQueued = true;
    requestAnimationFrame(applyGlowPosition);
  }
});

function setStatus(state) {
  const allowSpeakerBargeIn = localStorage.getItem("bargeInHeadphones") === "true";
  if (state === "listening" && !audioSendAllowed) {
    sendBuffer = new Float32Array(0);
    resampleRemainder = new Float32Array(0);
  }
  audioSendAllowed =
    state === "listening" ||
    state === "calibrating" ||
    (allowSpeakerBargeIn && state === "speaking" && bargeInReady && !bargeInBlocked && Boolean(socket) && !livekitSession && !liveSession);
  // Drives the flat CSS avatar's mouth/ring/waveform styling (see style.css's
  // .bot.state-* rules) by replacing only the state-* class, so the has-3d
  // class bot3d.js adds (see bot3d.js) survives every status change.
  if (botAvatar) {
    botAvatar.className = botAvatar.className.replace(/\bstate-\S+/, "").trim();
    botAvatar.classList.add(`state-${state}`);
  }
  // Drives the 3D avatar's ring color/speed and mouth idle shape; a no-op
  // if bot3d.js could not start (see its own WebGL feature check). The
  // word paced mouth animation itself is started per sentence by the
  // speaking_sentence event (see handleServerMessage), which the backend
  // sends right as each sentence's audio begins.
  if (window.bot3d) {
    window.bot3d.setState(state);
  }
  const avatar = newAvatar();
  if (avatar) avatar.setState(state);
}

function clearHint(container) {
  const hint = (container || transcript).querySelector(".hint");
  if (hint) hint.remove();
}

function addBubble(role, text, meta, container) {
  const target = container || transcript;
  clearHint(target);
  const bubble = document.createElement("div");
  bubble.className = `bubble bubble-${role}`;
  bubble.textContent = text;
  if (meta) {
    const metaSpan = document.createElement("span");
    metaSpan.className = "bubble-meta";
    metaSpan.textContent = meta;
    bubble.appendChild(metaSpan);
  }
  target.appendChild(bubble);
  target.scrollTop = target.scrollHeight;
  return bubble;
}

// Captions: mirrors only the assistant's own current line near the avatar
// (see index.html's #captions-overlay), never the user's own live speech
// transcript, so this reads as subtitles for what the assistant is
// saying, not a two sided transcript. setCaption() is therefore only
// called from the handlers below that carry the assistant's own text
// (updateStreamingReply, finalizeStreamingReply, the greeting, and
// errors), not from updateOrAddPartial/finalizePartial, which handle the
// user's side and still update the full history in #transcript as
// before, just without touching the caption overlay.
const captionsTextEl = document.getElementById("captions-text");

function setCaption(text) {
  if (captionsTextEl) captionsTextEl.textContent = text || "";
}

function updateOrAddPartial(text, language) {
  clearHint();
  if (!currentPartialBubble) {
    currentPartialBubble = document.createElement("div");
    currentPartialBubble.className = "bubble bubble-user bubble-partial";
    transcript.appendChild(currentPartialBubble);
  }
  currentPartialBubble.textContent = text;
  transcript.scrollTop = transcript.scrollHeight;
}

function finalizePartial(text) {
  if (currentPartialBubble) {
    currentPartialBubble.remove();
    currentPartialBubble = null;
  }
  if (text) {
    addBubble("user", text);
  }
}

// Mirrors updateOrAddPartial()/finalizePartial() above, but for the
// agent's own reply streaming in (see agent_response_chunk in
// handleServerMessage), so the reply visibly appears while the backend
// is still generating it (during the "thinking" status) rather than only
// once it is fully ready.
let currentStreamingReplyBubble = null;

function updateStreamingReply(text) {
  clearHint();
  if (!currentStreamingReplyBubble) {
    currentStreamingReplyBubble = document.createElement("div");
    currentStreamingReplyBubble.className = "bubble bubble-assistant bubble-partial";
    transcript.appendChild(currentStreamingReplyBubble);
  }
  currentStreamingReplyBubble.textContent = text;
  transcript.scrollTop = transcript.scrollHeight;
  setCaption(text);
}

function finalizeStreamingReply(text, mode) {
  if (currentStreamingReplyBubble) {
    currentStreamingReplyBubble.remove();
    currentStreamingReplyBubble = null;
  }
  addBubble("assistant", text, modeLabel(mode));
  setCaption(text);
}

// Simple linear interpolation resampler: good enough for speech, no
// external library needed for a demo running from a plain HTML page.
function resample(buffer, fromRate, toRate) {
  if (fromRate === toRate) return buffer;
  const ratio = fromRate / toRate;
  const newLength = Math.floor(buffer.length / ratio);
  const result = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    const srcIndex = i * ratio;
    const srcFloor = Math.floor(srcIndex);
    const frac = srcIndex - srcFloor;
    const s0 = buffer[srcFloor] || 0;
    const s1 = buffer[srcFloor + 1] || 0;
    result[i] = s0 + (s1 - s0) * frac;
  }
  return result;
}

function concatFloat32(a, b) {
  const result = new Float32Array(a.length + b.length);
  result.set(a, 0);
  result.set(b, a.length);
  return result;
}

function handleAudioProcess(event) {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;

  const inputData = event.inputBuffer.getChannelData(0);
  const nativeRate = audioContext.sampleRate;

  const combined = concatFloat32(resampleRemainder, inputData);
  const resampled = resample(combined, nativeRate, SAMPLE_RATE);
  resampleRemainder = new Float32Array(0);

  sendBuffer = concatFloat32(sendBuffer, resampled);

  while (sendBuffer.length >= CHUNK_SAMPLES) {
    const chunk = sendBuffer.slice(0, CHUNK_SAMPLES);
    sendBuffer = sendBuffer.slice(CHUNK_SAMPLES);

    // Captured audio is always removed from sendBuffer above regardless
    // (so it cannot grow unbounded while paused), but only actually sent
    // while audioSendAllowed is true and the user has not muted
    // themselves; see where audioSendAllowed is set, in setStatus(), for
    // why. This chunk is silently dropped, not queued for later, while
    // the assistant is thinking, speaking, or the mic is muted.
    if (audioSendAllowed && !isMuted) {
      socket.send(chunk.buffer);
    }

    if (!calibrationDone) {
      calibrationSamplesSent += CHUNK_SAMPLES;
      if (calibrationSamplesSent >= CALIBRATION_SECONDS * SAMPLE_RATE) {
        calibrationDone = true;
      }
    }
  }
}

function handleServerMessage(event) {
  let payload;
  try {
    payload = JSON.parse(event.data);
  } catch (err) {
    return;
  }

  switch (payload.type) {
    case "status":
      setStatus(payload.state);
      // While the hub is being asked the server adds a short line such as
      // "Checking your balance"; it clears as soon as the state changes.
      if (window.SLTConfirm) SLTConfirm.setStatus(payload.state === "thinking" && payload.detail ? payload.detail : null);
      break;
    case "confirmation":
      // A purchase or fault report waits for the customer's answer (modal).
      if (window.SLTConfirm && payload.confirmation) SLTConfirm.show(payload.confirmation);
      break;
    case "partial_transcript":
      updateOrAddPartial(payload.text, payload.language);
      break;
    case "final_transcript":
      finalizePartial(payload.text);
      rememberTurn("Customer", payload.text);
      sessionTurns.voice += 1;
      bargeInReady = true;
      break;
    case "agent_response_chunk":
      updateStreamingReply(payload.text);
      break;
    case "agent_response":
      bargeInBlocked = payload.mode === "closing" || payload.mode === "Introduction";
      finalizeStreamingReply(payload.text, payload.mode);
      if (window.SLTConfirm) SLTConfirm.setStatus(null);
      if (window.SltUi && window.SltUi.dashboard && payload.mode) {
        try { window.SltUi.dashboard.setMode(modeLabel(payload.mode)); } catch (err) { /* decoration only */ }
      }
      break;
    case "speaking_sentence":
      // The reply is now spoken while it is still being generated, so the
      // full text may not exist yet when audio starts; animating one
      // sentence at a time, as each one actually starts playing, keeps
      // the mouth in step with the real audio.
      if (window.bot3d) window.bot3d.speakText(payload.text);
      {
        // Over /ws/voice the server plays the audio, so the new avatar can
        // only follow the sentence text. Over LiveKit it follows the real
        // audio track instead (see attachAudioStream below).
        const avatar = newAvatar();
        if (avatar && !livekitSession) avatar.speakSentence(payload.text);
      }
      break;
    case "speaking_done":
      if (window.bot3d) window.bot3d.stopSpeaking();
      break;
    case "interrupt":
      // Barge in over LiveKit: the agent has already stopped its audio, so
      // only the mouth and the spoken sentence have to stop here.
      if (window.bot3d) window.bot3d.stopSpeaking();
      {
        const avatar = newAvatar();
        if (avatar) avatar.stop();
      }
      // The reply that was cut off stays on screen as it was when it stopped.
      if (currentStreamingReplyBubble) {
        currentStreamingReplyBubble.classList.remove("bubble-partial");
        currentStreamingReplyBubble = null;
      }
      break;
    case "user_name":
      // The user introduced themselves by voice ("my name is ...", see
      // _extract_user_name in backend/server.py); remember it the same way
      // a name typed on the user screen is remembered.
      userName = payload.name;
      try {
        localStorage.setItem("userName", payload.name);
      } catch (err) {
        /* localStorage unavailable; the name just will not be remembered next visit */
      }
      break;
    case "end_chat":
      // The backend has just spoken its closing goodbye (see
      // _sounds_like_closing_signal/_CLOSING_GOODBYE_TEXT in
      // backend/server.py) and is about to close the connection on its
      // own; endVoiceChat() below tears the session down the same way a
      // manual cut button press does, so both paths land in one place.
      endVoiceChat();
      break;
    case "error":
      setStatus("error");
      addBubble("assistant", tr("error.prefix", { message: payload.message }));
      setCaption(tr("error.prefix", { message: payload.message }));
      break;
    default:
      break;
  }
}

// Mic permission modal: shown once on entering the voice screen, before
// the browser's own native permission prompt, so a first time visitor has
// context for why that prompt is about to appear rather than being
// surprised by an unexplained browser popup. Clicking "Allow microphone"
// runs beginVoiceSession() below: the real getUserMedia() call (and, with
// it, the actual browser prompt) happens as part of startConversation(),
// and the backend speaks its introduction as soon as it connects. Dismissing this modal
// just hides it; the mute button still starts the conversation later if
// someone dismisses this now (see its own click handler).
const micPermissionModal = document.getElementById("mic-permission-modal");
const allowMicButton = document.getElementById("allow-mic-button");
const dismissMicModalButton = document.getElementById("dismiss-mic-modal");

async function showMicPermissionModal() {
  if (!micPermissionModal) return;
  // Best effort: skip the explanatory popup entirely, and go straight to
  // introducing itself and starting, if the browser already reports
  // microphone access as granted, since there is nothing left to explain
  // or ask permission for. Not every browser supports querying the
  // "microphone" permission name, so this silently falls through to
  // showing the modal if the check itself fails.
  try {
    if (navigator.permissions && navigator.permissions.query) {
      const status = await navigator.permissions.query({ name: "microphone" });
      if (status.state === "granted") {
        beginVoiceSession();
        return;
      }
    }
  } catch (err) {
    /* Permissions API unsupported for "microphone" in this browser; show the modal as normal */
  }
  micPermissionModal.classList.remove("hidden");
}

function hideMicPermissionModal() {
  if (micPermissionModal) micPermissionModal.classList.add("hidden");
}

if (allowMicButton) {
  allowMicButton.addEventListener("click", () => {
    hideMicPermissionModal();
    beginVoiceSession();
  });
}

if (dismissMicModalButton) {
  dismissMicModalButton.addEventListener("click", hideMicPermissionModal);
}

// Starts the microphone and backend connection straight away. Text to
// speech is on by default: the backend speaks its introduction out loud,
// addressing the visitor by name or as Guest, as the very first thing in
// the session, then calibrates and starts listening (see voice_ws in
// backend/server.py). greetingsGiven.voice asks it to skip the
// introduction when reconnecting later in the same visit (the mic button
// after a dropped connection), so it only introduces itself once.
async function beginVoiceSession() {
  await startConversation(!greetingsGiven.voice);
  // Only counts as greeted once the session actually started; if the mic
  // was denied, the next attempt should still introduce itself.
  if (isActive) greetingsGiven.voice = true;
}

async function startConversation(greet = false, handover = false) {
  if (isActive || isConnecting) return;
  isConnecting = true;

  // Tier 1, Gemini Live through the backend, when the voice router says it is up.
  // Any failure falls through to LiveKit and then the local pipeline below.
  if (await tryStartGeminiLive(greet, handover)) return;

  // The LiveKit transport, when the server is set to it. Any failure falls
  // through to the original WebSocket path below, which is unchanged.
  if (await tryStartLiveKit(handover)) return;

  const backendUrl = backendWsUrl();
  if (!backendUrl) {
    setCaption(tr("error.unreachable"));
    isConnecting = false;
    return;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    bargeInReady = false;
    bargeInBlocked = false;
  } catch (err) {
    setCaption(tr("voice.mic_denied"));
    setStatus("error");
    isConnecting = false;
    return;
  }

  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  sourceNode = audioContext.createMediaStreamSource(mediaStream);
  // ScriptProcessorNode is deprecated in favor of AudioWorklet, but needs
  // no separate worklet module file to load, which keeps this a single
  // page demo; a production rewrite should move this to AudioWorklet.
  processorNode = audioContext.createScriptProcessor(4096, 1, 1);
  processorNode.onaudioprocess = handleAudioProcess;
  sourceNode.connect(processorNode);
  processorNode.connect(audioContext.destination);

  resampleRemainder = new Float32Array(0);
  sendBuffer = new Float32Array(0);
  calibrationSamplesSent = 0;
  calibrationDone = false;

  socket = new WebSocket(backendUrl);
  socket.binaryType = "arraybuffer";

  socket.onopen = () => {
    // session_token is null for a Guest; for a verified user the backend
    // takes the name from the account it identifies, not from user_name.
    socket.send(
      JSON.stringify({
        type: "start_session",
        language_hint: selectedLanguage,
        user_name: userName,
        greet,
        handover,
        session_token: sessionToken,
        voice_style: voiceStyle(),
      })
    );
  };

  socket.onmessage = handleServerMessage;

  socket.onclose = () => {
    stopConversation();
  };

  socket.onerror = () => {
    setCaption(tr("error.unreachable"));
    setStatus("error");
  };

  isActive = true;
  isConnecting = false;
  isMuted = false;
  updateMicButtonUI();
}

// Tier 1: Gemini Live through /ws/live. Returns true once the conversation runs on it,
// false to let the caller use the next tier. A failure after it started (the server
// already told the router) says a short notice and continues on the next tier.
async function tryStartGeminiLive(greet, handover) {
  const live = window.LiveVoice;
  if (!live) return false;
  // Right after a Gemini Live failure the next tier is started without asking the router,
  // which may not have been told yet.
  if (skipLiveOnce) {
    skipLiveOnce = false;
    return false;
  }
  try {
    const status = await live.tier(backendHttpBase());
    if (status.tier !== "gemini_live") return false;
    const wsUrl = backendWsUrl();
    if (!wsUrl) return false;
    setCaption(tr("loader.connecting"));
    const model = status.model || "";
    liveSession = await live.connect({
      wsUrl: wsUrl.replace(/\/ws\/voice$/, "/ws/live"),
      language: selectedLanguage,
      voiceStyle: voiceStyle(),
      userName: userName,
      sessionToken: sessionToken,
      greet: greet && !handover,
      context: handover ? handoverContext() : "",
      onEvent: (payload) => {
        if (payload.type === "agent_response") {
          rememberTurn("Assistant", payload.text);
          payload = Object.assign({}, payload, { mode: `gemini live (${model})` });
        }
        handleServerMessage({ data: JSON.stringify(payload) });
      },
      onAudio: (analyser) => {
        const bridge = window.avatarBridge;
        if (bridge && bridge.speakAudio) bridge.speakAudio(analyser, null, { language: selectedLanguage });
      },
      onFailed: () => handoverFromLive(),
      onClosed: () => {
        if (liveSession) stopConversation();
      },
    });
  } catch (err) {
    console.warn("Gemini Live unavailable, using the next voice tier:", err && err.reason ? err.reason : err);
    liveSession = null;
    return false;
  }
  isActive = true;
  isConnecting = false;
  isMuted = false;
  audioSendAllowed = true;
  updateMicButtonUI();
  return true;
}

// Tier 1 failed in the middle of a conversation: say so briefly, then start the next
// tier with the same language, voice, sign in and the last lines of the conversation.
let skipLiveOnce = false;
function handoverFromLive() {
  liveSession = null;
  skipLiveOnce = true;
  stopConversation();
  setCaption(tr("voice.reconnecting"));
  setTimeout(() => startConversation(false, true), 300);
}

// Joins a LiveKit room instead of opening /ws/voice. Returns true once the
// conversation is running on it, false to let the caller use the WebSocket.
async function tryStartLiveKit(handover = false) {
  const livekit = window.LiveKitVoice;
  if (!livekit) return false;
  try {
    if (!(await livekit.isEnabled(backendHttpBase()))) return false;
    setCaption(tr("loader.connecting"));
    livekitSession = await livekit.connect({
      baseUrl: backendHttpBase(),
      language: selectedLanguage,
      voiceStyle: voiceStyle(),
      userName: userName,
      sessionToken: sessionToken,
      handover: handover,
      onEvent: (payload) => handleServerMessage({ data: JSON.stringify(payload) }),
      onAgentAudio: (track) => {
        const avatar = newAvatar();
        if (avatar) avatar.attachAudioStream(new MediaStream([track.mediaStreamTrack]));
      },
      onNeedsGesture: () => setCaption(tr("voice.tap_sound")),
      onDisconnected: () => {
        if (livekitSession) stopConversation();
      },
    });
  } catch (err) {
    console.warn("LiveKit transport unavailable, using the WebSocket:", err);
    if (livekitSession) {
      try {
        livekitSession.disconnect();
      } catch (e) {
        /* already closed */
      }
    }
    livekitSession = null;
    return false;
  }
  isActive = true;
  isConnecting = false;
  isMuted = false;
  updateMicButtonUI();
  return true;
}

function stopConversation() {
  const closingLive = liveSession;
  liveSession = null;
  if (closingLive) {
    try {
      closingLive.disconnect();
    } catch (err) {
      /* already closed */
    }
  }
  const closingRoom = livekitSession;
  livekitSession = null;
  if (closingRoom) {
    try {
      closingRoom.disconnect();
    } catch (err) {
      /* already closed */
    }
  }
  const avatar = newAvatar();
  if (avatar) avatar.stop();
  isActive = false;
  isConnecting = false;
  isMuted = false;
  audioSendAllowed = false;
  setStatus("idle");
  updateMicButtonUI();

  if (socket && socket.readyState === WebSocket.OPEN) {
    try {
      socket.send(JSON.stringify({ type: "end_session" }));
    } catch (err) {
      /* socket already closing */
    }
    socket.close();
  }
  socket = null;

  if (processorNode) {
    processorNode.disconnect();
    processorNode = null;
  }
  if (sourceNode) {
    sourceNode.disconnect();
    sourceNode = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  if (currentPartialBubble) {
    currentPartialBubble.remove();
    currentPartialBubble = null;
  }
  if (currentStreamingReplyBubble) {
    currentStreamingReplyBubble.remove();
    currentStreamingReplyBubble = null;
  }
}

// Mic button doubles as the conversation's only manual start/mute control:
// if the conversation has not started yet (someone dismissed the mic
// permission modal, or the WebSocket dropped), clicking it runs the exact
// same beginVoiceSession() the modal's own Allow button runs; once active,
// it toggles isMuted instead, which pauses sending audio (see
// handleAudioProcess()) without tearing down the connection. Ending the
// conversation outright is the separate end-chat button below, not this one.
const micToggleButton = document.getElementById("mic-toggle");
const micIcon = document.getElementById("mic-icon");

const ICON_MIC_INNER =
  '<rect x="9" y="2" width="6" height="12" rx="3"></rect><path d="M5 10v1a7 7 0 0 0 14 0v-1"></path>' +
  '<line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line>';
const ICON_MIC_OFF_INNER =
  '<line x1="1" y1="1" x2="23" y2="23"></line>' +
  '<path d="M9 9v3a3 3 0 0 0 5.12 2.12M15 9.34V4a3 3 0 0 0-5.94-.6"></path>' +
  '<path d="M17 16.95A7 7 0 0 1 5 12v-2"></path><path d="M19 10v2a7 7 0 0 1-.11 1.23"></path>' +
  '<line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line>';

function updateMicButtonUI() {
  if (!micToggleButton) return;
  if (!isActive) {
    micToggleButton.setAttribute("aria-pressed", "false");
    micToggleButton.setAttribute("aria-label", tr("voice.mic.start"));
    if (micIcon) micIcon.innerHTML = ICON_MIC_INNER;
  } else if (isMuted) {
    micToggleButton.setAttribute("aria-pressed", "true");
    micToggleButton.setAttribute("aria-label", tr("voice.mic.unmute"));
    if (micIcon) micIcon.innerHTML = ICON_MIC_OFF_INNER;
  } else {
    micToggleButton.setAttribute("aria-pressed", "false");
    micToggleButton.setAttribute("aria-label", tr("voice.mic.mute"));
    if (micIcon) micIcon.innerHTML = ICON_MIC_INNER;
  }
}

if (micToggleButton) {
  micToggleButton.addEventListener("click", () => {
    if (!isActive) {
      hideMicPermissionModal();
      beginVoiceSession();
      return;
    }
    isMuted = !isMuted;
    updateMicButtonUI();
    if (livekitSession) livekitSession.setMuted(isMuted).catch(() => {});
  });
}

// Ends the conversation outright, unlike mic-toggle's mute (which just
// pauses sending audio). Reused for both a manual tap here and the
// backend's own end_chat event (see handleServerMessage above), so a
// conversation that ends itself, once the user sounds satisfied (see
// _sounds_like_closing_signal in backend/server.py), behaves identically
// to one the user ends by hand.
const endChatButton = document.getElementById("end-chat-button");

function endVoiceChat() {
  stopConversation();
  greetingsGiven.voice = false;
  navigateTo("select");
  askForRating("voice");
}

// Leaving the text chat through its back link ends that chat.
const chatBackLink = document.querySelector("#screen-chat .back-link");
if (chatBackLink) {
  chatBackLink.addEventListener("click", () => askForRating("chat"));
}

if (endChatButton) {
  endChatButton.addEventListener("click", endVoiceChat);
}

// Logout: ends a verified user's session on the backend (clears its
// cookie), forgets the name and token here, and sends the visitor back to
// the very first screen. For a Guest there is no session to end.
const logoutButton = document.getElementById("logout-button");

if (logoutButton) {
  logoutButton.addEventListener("click", () => {
    stopConversation();
    fetch(`${backendHttpBase()}/verify/logout`, {
      method: "POST",
      credentials: "include",
      headers: authHeaders(),
    }).catch(() => {
      /* backend unreachable; the short lived session simply expires */
    });
    if (sessionToken) {
      fetch(`${backendHttpBase()}/auth/logout`, {
        method: "POST",
        credentials: "include",
        headers: authHeaders(),
      }).catch(() => {
        /* backend unreachable; the short lived session simply expires */
      });
    }
    sessionToken = null;
    userName = null;
    greetingsGiven.voice = false;
    greetingsGiven.chat = false;
    try {
      localStorage.removeItem("userName");
    } catch (err) {
      /* localStorage unavailable; nothing to clear */
    }
    navigateTo("language");
    askForRating("chat");
    askForRating("voice");
  });
}

// Male/female switch: changes the avatar and the voice together, takes effect
// mid conversation from the next sentence, and is remembered in the browser.
// Over /ws/voice the server learns it from a set_voice message, over LiveKit
// from the same message on the control data channel.
const genderToggleButton = document.getElementById("gender-toggle");
const genderLetterEl = document.getElementById("gender-toggle-letter");

function applyGenderUI() {
  if (!genderToggleButton) return;
  const isMale = avatarGender === "male";
  genderToggleButton.setAttribute("aria-pressed", String(isMale));
  genderToggleButton.setAttribute(
    "aria-label",
    tr(isMale ? "voice.gender.male" : "voice.gender.female")
  );
  if (genderLetterEl) genderLetterEl.textContent = tr(isMale ? "voice.gender.letter_male" : "voice.gender.letter_female");
}

function setAvatarGender(gender) {
  avatarGender = gender === "male" ? "male" : "female";
  try {
    localStorage.setItem("avatarGender", avatarGender);
  } catch (err) {
    /* localStorage unavailable; the choice just will not be remembered */
  }
  applyGenderUI();
  if (window.avatarBridge) window.avatarBridge.setGender(avatarGender);
  if (liveSession) {
    // A Gemini Live voice is fixed when the session opens: reconnect with the new one,
    // carrying the conversation so far, and no second introduction.
    stopConversation();
    setTimeout(() => startConversation(false, true), 200);
  } else if (livekitSession) {
    livekitSession.setVoice(voiceStyle()).catch(() => {});
  } else if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "set_voice", voice_style: voiceStyle() }));
  }
}

// The new avatar has no male model file yet: it keeps the female face, so say
// so rather than let the switch look broken. The voice still changes.
document.addEventListener("avatar:gender-fallback", (event) => {
  const reason = event.detail && event.detail.message;
  setCaption(tr("voice.male_missing"));
});

if (genderToggleButton) {
  applyGenderUI();
  // The saved choice also decides which avatar the 3D renderer starts with (the bridge remembers
  // it until the renderer is ready); before this the voice was male while the face was female.
  if (window.avatarBridge) window.avatarBridge.setGender(avatarGender);
  genderToggleButton.addEventListener("click", () => {
    setAvatarGender(avatarGender === "male" ? "female" : "male");
  });
}

// Captions on/off: toggles only the live overlay's visibility; the full
// transcript in the history panel keeps accumulating either way, since
// captions and history are deliberately separate concerns (see
// index.html's comments on #captions-overlay and #history-panel).
const captionsToggleButton = document.getElementById("captions-toggle");
const captionsOverlay = document.getElementById("captions-overlay");

if (captionsToggleButton) {
  captionsToggleButton.addEventListener("click", () => {
    const nowVisible = captionsToggleButton.getAttribute("aria-pressed") !== "true";
    captionsToggleButton.setAttribute("aria-pressed", String(nowVisible));
    if (captionsOverlay) captionsOverlay.classList.toggle("hidden", !nowVisible);
  });
}

// Conversation history: a simple show/hide overlay over the same
// #transcript element every screen already renders bubbles into, so
// opening it costs nothing beyond toggling one class.
const historyToggleButton = document.getElementById("history-toggle");
const historyPanel = document.getElementById("history-panel");
const closeHistoryButton = document.getElementById("close-history");

if (historyToggleButton) {
  historyToggleButton.addEventListener("click", () => {
    if (historyPanel) historyPanel.classList.remove("hidden");
  });
}

if (closeHistoryButton) {
  closeHistoryButton.addEventListener("click", () => {
    if (historyPanel) historyPanel.classList.add("hidden");
  });
}

// Text chat screen: a lightweight alternative to the voice flow, talking
// to the backend's POST /chat endpoint (see backend/server.py), which
// runs the same Orchestrator used by the voice pipeline, just without any
// STT or TTS involved. Derives the chat endpoint from the same backend
// field the voice screen uses, so there is only one place to configure
// the backend location.
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatTranscript = document.getElementById("chat-transcript");
const chatModeBadge = document.getElementById("chat-mode-badge");
let lastChatMode = null;

function chatHttpUrl() {
  return `${backendHttpBase()}/chat`;
}

function setChatModeBadge(mode) {
  lastChatMode = mode;
  if (!chatModeBadge) return;
  if (!mode) {
    chatModeBadge.textContent = tr("chat.mode_none");
    chatModeBadge.className = "badge badge-muted";
    return;
  }
  const isCloud = mode.toLowerCase().includes("cloud");
  chatModeBadge.textContent = tr("chat.mode", { mode: modeLabel(mode) });
  chatModeBadge.className = `badge ${isCloud ? "badge-cloud" : "badge-local"}`;
}

if (chatForm) {
  chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = chatInput.value.trim();
    if (!text) return;

    addBubble("user", text, null, chatTranscript);
    sessionTurns.chat += 1;
    chatInput.value = "";
    chatInput.disabled = true;

    // Three pulsing dots while the answer is on its way.
    const typingBubble = addBubble("assistant", "", null, chatTranscript);
    typingBubble.classList.add("bubble-typing");
    for (let i = 0; i < 3; i++) {
      const dot = document.createElement("span");
      dot.className = "dot";
      typingBubble.appendChild(dot);
    }

    try {
      const response = await fetch(chatHttpUrl(), {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({ text, language: selectedLanguage, user_name: userName }),
      });
      if (!response.ok) {
        throw new Error(`Backend returned ${response.status}`);
      }
      const data = await response.json();
      typingBubble.remove();
      addBubble("assistant", data.reply, modeLabel(data.mode), chatTranscript);
      setChatModeBadge(data.mode);
      if (window.SLTConfirm) {
        if (data.confirmation) SLTConfirm.show(data.confirmation);
        else SLTConfirm.hide();
      }
    } catch (err) {
      typingBubble.remove();
      addBubble("assistant", tr("error.unreachable"), null, chatTranscript);
    } finally {
      chatInput.disabled = false;
      chatInput.focus();
    }
  });
}

// The answer to a confirmation (executed, cancelled, expired or failed) joins
// the transcript with the hub's own wording. See ui/confirm/confirm.js.
document.addEventListener("slt:confirmation-result", function (event) {
  var detail = (event && event.detail) || {};
  if (detail.text) addBubble("assistant", detail.text, null, chatTranscript);
});

// Maps the English mode text the backend sends to a translated label.
function modeLabel(mode) {
  if (!mode) return "";
  const text = String(mode);
  const lower = text.toLowerCase();
  let label;
  if (lower.startsWith("cloud")) label = tr("mode.cloud");
  else if (lower.startsWith("local")) label = tr("mode.local_agent");
  else if (lower.startsWith("access policy")) label = tr("mode.policy");
  else if (lower.startsWith("slt hub")) label = tr("mode.hub");
  else if (lower.startsWith("account data unavailable")) label = tr("mode.account_unavailable");
  else if (lower.startsWith("gemini live")) label = tr("voice.mode.gemini_live", { model: (text.match(/\(([^)]+)\)/) || [])[1] || "" });
  else if (lower.startsWith("livekit")) label = tr("voice.mode.livekit");
  else if (lower.startsWith("local voice")) label = tr("voice.mode.local");
  else return text;
  if (/\+ SLT account data/i.test(text)) label = tr("mode.plus_account", { mode: label });
  return label;
}

function repaintDynamicLabels() {
  updateMicButtonUI();
  applyGenderUI();
  setChatModeBadge(lastChatMode);
  setOtpMode(otpMode);
}
document.addEventListener("slt:language", repaintDynamicLabels);
