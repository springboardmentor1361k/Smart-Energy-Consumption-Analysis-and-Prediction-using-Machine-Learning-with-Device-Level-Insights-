console.log('Frontend loaded');

// Simulate live updates
setInterval(() => {
    const powerEls = document.querySelectorAll('.stat-value');
    if(powerEls.length > 0) {
        const current = parseFloat(powerEls[0].textContent);
        const variation = Math.random() > 0.5 ? 10 : -10;
        powerEls[0].textContent = Math.round(current + variation);
    }
}, 3000);
