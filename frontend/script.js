document.getElementById("fraudForm").addEventListener("submit", async function (e) {
    e.preventDefault();

    const form = e.target;
    const btn = document.getElementById("submitBtn");

    const isoTime = new Date(form.transaction_time.value).toISOString();

    const data = {
        account_age_days: parseInt(form.account_age_days.value),
        total_transactions_user: parseInt(form.total_transactions_user.value),
        avg_amount_user: parseFloat(form.avg_amount_user.value),
        amount: parseFloat(form.amount.value),
        channel: form.channel.value,
        merchant_category: form.merchant_category.value,
        promo_used: parseInt(form.promo_used.value),
        avs_match: parseInt(form.avs_match.value),
        cvv_result: parseInt(form.cvv_result.value),
        three_ds_flag: parseInt(form.three_ds_flag.value),
        transaction_time: isoTime,
        shipping_distance_km: parseFloat(form.shipping_distance_km.value),
        country: form.country.value.toUpperCase(),
        bin_country: form.bin_country.value.toUpperCase(),
    };

    btn.textContent = "Running…";
    btn.disabled = true;

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
        });
        if (!response.ok) throw new Error("API error");
        const result = await response.json();
        renderResult(result);
    } catch (err) {
        renderError();
    } finally {
        btn.textContent = "Run check";
        btn.disabled = false;
    }
});

function renderResult(result) {
    const card = document.getElementById("resultCard");
    card.className = "result-card show";

    const level = result.risk_level;
    const colorMap = { Low: "#15803D", Medium: "#D97706", High: "#DC2626" };
    const classMap = { Low: "low", Medium: "medium", High: "high" };

    const badge = document.getElementById("resultBadge");
    badge.textContent = level + " risk";
    badge.className = "result-badge " + classMap[level];

    const prob = result.fraud_probability * 100;
    document.getElementById("resultProb").textContent = prob.toFixed(2) + "%";

    const bar = document.getElementById("resultBar");
    bar.style.width = Math.min(prob, 100) + "%";
    bar.style.background = colorMap[level] || "#D97706";
}

function renderError() {
    const card = document.getElementById("resultCard");
    card.className = "result-card show";
    document.getElementById("resultBadge").textContent = "Error";
    document.getElementById("resultBadge").className = "result-badge high";
    document.getElementById("resultProb").textContent = "—";
    const bar = document.getElementById("resultBar");
    bar.style.width = "0%";
}