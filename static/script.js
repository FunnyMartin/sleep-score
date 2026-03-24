var jsDay = new Date().getDay();
var pyDay = jsDay === 0 ? 6 : jsDay - 1;
document.getElementById("dow").value = pyDay;


function submitForm() {
    clearError();
    document.getElementById("result").style.display = "none";

    var hrValues = [];
    for (var i = 1; i <= 7; i++) {
        var val = document.getElementById("hr" + i).value.trim();
        if (val === "") {
            showError("Vyplň HR pro noc " + i + ".");
            return;
        }
        var num = parseFloat(val);
        if (isNaN(num) || num < 30 || num > 130) {
            showError("Noc " + i + ": zadej HR mezi 30 a 130 BPM.");
            return;
        }
        hrValues.push(num);
    }

    var stepsInput = document.getElementById("steps").value.trim();
    var kcalInput  = document.getElementById("kcal").value.trim();
    var steps = stepsInput === "" ? 0 : parseFloat(stepsInput);
    var kcal  = kcalInput  === "" ? 0 : parseFloat(kcalInput);
    var dow   = parseInt(document.getElementById("dow").value);

    var btn = document.getElementById("submit-btn");
    btn.disabled = true;
    btn.textContent = "počítám...";

    var payload = {
        hr1: hrValues[0], hr2: hrValues[1], hr3: hrValues[2],
        hr4: hrValues[3], hr5: hrValues[4], hr6: hrValues[5],
        hr7: hrValues[6],
        steps: steps,
        kcal: kcal,
        dow: dow
    };

    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    })
        .then(function(response) { return response.json(); })
        .then(function(data) {
            btn.disabled = false;
            btn.textContent = "spustit predikci";

            if (data.error) {
                showError(data.error);
                return;
            }

            var resultDiv = document.getElementById("result");
            var cls = data.prediction === 1 ? "good" : "bad";
            resultDiv.className = cls;
            resultDiv.style.display = "block";

            document.getElementById("result-title").textContent = data.label;
            document.getElementById("result-prob").textContent =
                "pravděpodobnost dobré noci: " + data.probability + "% (práh: " + data.threshold + "%)";

            var fill = document.getElementById("prob-fill");
            fill.style.width = data.probability + "%";
        })
        .catch(function() {
            btn.disabled = false;
            btn.textContent = "spustit predikci";
            showError("Chyba připojení k serveru.");
        });
}

function showError(msg) {
    var box = document.getElementById("error-box");
    box.textContent = msg;
    box.style.display = "block";
}

function clearError() {
    document.getElementById("error-box").style.display = "none";
}