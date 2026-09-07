document.addEventListener("DOMContentLoaded", function () {

    // ===== Prediction =====
    const form = document.getElementById("predictionForm");
    const resultDiv = document.getElementById("predictionResult");

    form.addEventListener("submit", function (e) {
        e.preventDefault();

        const hour = document.getElementById("hour").value;
        const day = document.getElementById("day").value;
        const month = document.getElementById("month").value;

        fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                hour: hour,
                day: day,
                month: month
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.prediction !== undefined) {
                resultDiv.innerHTML =
                    "Predicted Energy Consumption: <b>" +
                    data.prediction.toFixed(3) +
                    " kW</b>";
            } else {
                resultDiv.innerHTML = "Prediction failed.";
            }
        })
        .catch(() => {
            resultDiv.innerHTML = "Prediction failed.";
        });
    });


    // ===== Charts =====
    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: { color: "#ffffff" }
            }
        },
        scales: {
            x: {
                ticks: { color: "#ffffff" }
            },
            y: {
                ticks: { color: "#ffffff" }
            }
        }
    };

    // Hourly
    new Chart(document.getElementById("hourlyChart"), {
        type: "line",
        data: {
            labels: ["1AM","4AM","8AM","12PM","4PM","8PM","11PM"],
            datasets: [{
                label: "Hourly Consumption (kW)",
                data: [1.2, 1.0, 1.8, 2.5, 2.2, 2.8, 1.5],
                borderColor: "#00e0ff",
                tension: 0.4
            }]
        },
        options: chartOptions
    });

    // Daily
    new Chart(document.getElementById("dailyChart"), {
        type: "bar",
        data: {
            labels: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
            datasets: [{
                label: "Daily Usage (kWh)",
                data: [18,20,19,22,24,26,21],
                backgroundColor: "#00e0ff"
            }]
        },
        options: chartOptions
    });

    // Weekly
    new Chart(document.getElementById("weeklyChart"), {
        type: "line",
        data: {
            labels: ["Week 1","Week 2","Week 3","Week 4"],
            datasets: [{
                label: "Weekly Usage (kWh)",
                data: [140,160,150,170],
                borderColor: "#00e0ff",
                tension: 0.4
            }]
        },
        options: chartOptions
    });

});
