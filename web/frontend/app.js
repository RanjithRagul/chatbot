const form = document.querySelector("#chat-form");
const promptInput = document.querySelector("#prompt");
const messagesElement = document.querySelector("#messages");
const sendButton = document.querySelector("#send-button");
const serviceStatus = document.querySelector("#service-status");

const MAX_CONTEXT_MESSAGES = 10;
const MAX_CONTEXT_CHARS = 12_000;
const conversation = [];
let waitingForAnswer = false;

function trimConversationForNextMessage(text) {
  let retainedCharacters = conversation.reduce(
    (total, message) => total + message.content.length,
    0,
  );

  while (
    conversation.length > MAX_CONTEXT_MESSAGES ||
    retainedCharacters + text.length > MAX_CONTEXT_CHARS
  ) {
    const removedMessages = conversation.splice(0, 2);
    if (removedMessages.length === 0) break;
    retainedCharacters -= removedMessages.reduce(
      (total, message) => total + message.content.length,
      0,
    );
  }
}

function addMessage(role, content, { pending = false } = {}) {
  const article = document.createElement("article");
  article.className = `message ${role === "user" ? "user-message" : "assistant-message"}`;
  if (pending) article.classList.add("pending");

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = "✦";

  const messageContent = document.createElement("div");
  messageContent.className = "message-content";
  const paragraph = document.createElement("p");
  paragraph.textContent = content;
  messageContent.append(paragraph);

  article.append(avatar, messageContent);
  messagesElement.append(article);
  article.scrollIntoView({ block: "nearest", behavior: "smooth" });
  return article;
}

function setServiceStatus(state, label) {
  serviceStatus.classList.toggle("is-online", state === "online");
  serviceStatus.classList.toggle("is-offline", state === "offline");
  serviceStatus.lastElementChild.textContent = label;
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health", { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error("Health check failed");
    const health = await response.json();
    if (!health.chat_configured) {
      setServiceStatus("offline", "Chat needs an API key");
      return;
    }
    setServiceStatus("online", "Service ready");
  } catch {
    setServiceStatus("offline", "Service unavailable");
  }
}

async function sendMessage(text) {
  trimConversationForNextMessage(text);
  const userMessage = { role: "user", content: text };
  conversation.push(userMessage);
  addMessage("user", text);
  const pendingMessage = addMessage("assistant", "Thinking…", { pending: true });
  waitingForAnswer = true;
  sendButton.disabled = true;
  promptInput.disabled = true;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ messages: conversation }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "The request failed.");
    if (typeof payload.answer !== "string" || !payload.answer.trim()) {
      throw new Error("The assistant returned an empty answer.");
    }

    conversation.push({ role: "assistant", content: payload.answer });
    pendingMessage.querySelector("p").textContent = payload.answer;
    pendingMessage.classList.remove("pending");
  } catch (error) {
    conversation.pop();
    pendingMessage.remove();
    addMessage("assistant", error.message || "Something went wrong. Please try again.");
  } finally {
    waitingForAnswer = false;
    sendButton.disabled = false;
    promptInput.disabled = false;
    promptInput.focus();
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (waitingForAnswer) return;

  const text = promptInput.value.trim();
  if (!text) return;

  promptInput.value = "";
  promptInput.style.height = "auto";
  await sendMessage(text);
});

promptInput.addEventListener("input", () => {
  promptInput.style.height = "auto";
  promptInput.style.height = `${Math.min(promptInput.scrollHeight, 190)}px`;
});

promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

checkHealth();
