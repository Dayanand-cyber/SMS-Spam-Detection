
const messageBox = document.getElementById("message");
const checkButton = document.getElementById("checkButton");
const resultBox = document.getElementById("result");

checkButton.addEventListener("click", async () => {
    const message = messageBox.value.trim();

    resultBox.className = "result";
    resultBox.style.display = "block";

    if (!message) {
        resultBox.textContent = "Please enter a message.";
        return;
    }

    checkButton.disabled = true;
    checkButton.textContent = "Checking...";
    resultBox.textContent = "Analyzing your message...";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message: message })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Something went wrong.");
        }

        if (data.prediction === "SPAM") {
            resultBox.className = "result spam";
            resultBox.textContent =
                `SPAM DETECTED — Model score: ${data.confidence}%`;
        } else {
            resultBox.className = "result ham";
            resultBox.textContent =
                `HAM — Legitimate message. Model score: ${data.confidence}%`;
        }
    } catch (error) {
        resultBox.className = "result";
        resultBox.style.display = "block";
        resultBox.textContent = error.message;
    } finally {
        checkButton.disabled = false;
        checkButton.textContent = "Check Message";
    }
});
