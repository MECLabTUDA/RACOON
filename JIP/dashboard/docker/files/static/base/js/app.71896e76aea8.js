var chartManager = new ChartManager();

window.onresize = function() {
    chartManager.handleWindowResize();
};