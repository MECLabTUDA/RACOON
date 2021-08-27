class Plot {

    constructor(element, height) {
        this.parentElement = element;
        this.$content = $(`<div style="height: ${height}rem;"></div>`);

        this.option = {};
        this.renderPlot();
    }

    // Shows a plot selected in the plots section of a POI Info modal
    async showPlotFromId(plot_id) {
        this.plotChart.showLoading();

        let url = `${ API_URL }/plotData/?plot_id=${plot_id}`;
        const res = await axios.get(url);
        // Apache Echarts options are returned by the API request
        let plotOptions = res.data[0].plot_data;

        this.plotChart.setOption(plotOptions);

        this.plotChart.hideLoading();
    }

    // Shows a plot that display the 90 day history of a given metric
    async showPlotFromHist(measure_id, measure_name, location_id, date, lookbackDays, titleText){
        this.plotChart.showLoading();
        var plotData = [];

        // Init Dates
        var toDate = new Date(date);
        var fromDate = new Date();
        fromDate.setDate(toDate.getDate() - lookbackDays - 30);
        
        var fromDate_thirty = new Date(toDate);
        fromDate_thirty.setDate(toDate.getDate() - 30);

        var formattedToDate = this.formatDate(toDate);
        var formattedFromDate = this.formatDate(fromDate);

        // Get data from API
        let apiResult = {};
        let url = `${ API_URL }/data/?location=${location_id}&date__gte=${formattedFromDate}&date__lte=${formattedToDate}`;
        const res = await axios.get(url);
        res.data.forEach(entry => {
            let present_measures = entry.measureData.length > 0 ? {...Object.assign(...entry.measureData.map((me) => ({[me.measure]: me.value})))} : {};
            let val = ($.inArray(measure_id, Object.keys(present_measures)) != -1) ? present_measures[measure_id] : 0;
            apiResult[entry.date] = isNaN(val) ? 0 : parseFloat(val.toFixed(DECIMALS));
        });

        // Prepare data for plotting, add missing dates to dict
        toDate.setDate(toDate.getDate() + 1);
        for (let day = fromDate; day < toDate; day.setDate(day.getDate() + 1)) {
            let dayString  = this.formatDate(day);
            if ($.inArray(dayString, Object.keys(apiResult)) != -1){
                plotData.push([new Date(dayString), apiResult[dayString]]);
            } else {
                plotData.push([new Date(dayString), 0]);
            }
        }

        // Apache Echarts definition of plot design
        var plotOptions = {
            backgroundColor: THEME === 'dark' ? '#333333' : '#FFFFFF', 
            title: {
                text: titleText,
                textStyle: {
                    color: '#A2A2A2',
                    fontWeight: 'normal',
                    fontSize: 13,
                },
                left: 'center',
                padding: 10,
            },
            animationDuration: 500,
            tooltip : {
                trigger: 'axis',
                formatter: function (params) {        
                    return `${formatDateForLangLong(params[0].data[0])}<br/>
                            ${params[0].marker} ${params[0].seriesName}: ${params[0].data[1]}`;
                }
            },
            grid: {
                left: 40,
                top: 32,
                right: 15,
                bottom: 68
            },
            dataZoom: [{
                type: 'slider',
                labelFormatter: (function(d){
                        var date = new Date(d);
                        return formatDateForLangLong(date);
                }),
                textStyle: {
                    fontSize: 10,
                },
                startValue: fromDate_thirty,
                endValue: toDate,
                right: 30,
                left: 30,
                dataBackground: {
                    lineStyle: {
                        color: '#EE7218',
                    },
                    areaStyle: {
                        color: '#EE7218',
                        opacity: 0.35,
                    }
                },
                selectedDataBackground: {
                    lineStyle: {
                        color: '#BAB6B3',
                    },
                    areaStyle: {
                        color: '#BAB6B3',
                    }
                },
                fillerColor:'rgba(164, 158, 152, 0.3)',
                handleStyle: {
                    color: '#f1904b',
                },
                brushSelect: false,
            }],
            calculable : true,
            xAxis : [
                {
                    type : 'time',
                    axisLabel: {
                        formatter: (function(d){
                            var date = new Date(d);
                            return formatDateForLangShort(date);
                        }),
                        fontSize: 10,
                        margin: 10,
                        rotate: 20,
                    },
                }
            ],
            yAxis : [
                {
                    type : 'value',
                }
            ],
            series : [
                {
                    color: '#eb8133',
                    name: measure_name,
                    type: 'line',
                    lineStyle: {
                        width: 3
                    },
                    itemStyle: {
                        normal: {
                            areaStyle: {
                                type: 'default'
                            }
                        }
                    },
                    data: plotData
                },
            ]
        }
        this.plotChart.setOption(plotOptions);

        this.plotChart.hideLoading();
    }

    async setPlotFromJson(options) {
        this.plotChart.showLoading();

        this.plotChart.setOption(options);

        this.plotChart.hideLoading();
    }

    renderPlot() {
        $(this.parentElement).empty();
        $(this.parentElement).append(this.$content);

        this.plotChart = echarts.init(this.$content[0], THEME);
        this.plotChart.setOption(this.option);

        setTimeout(() => this.resize(), 10);

        var pC = this.plotChart;
        window.onresize = function() {
            pC.resize();
        };
    }

    resize() {
        this.plotChart.resize();
    }

    formatDate(date) {
        return date.toISOString().slice(0,10);
    }    
}

// Date Formatter for compact View. e.g. 10.11.
function formatDateForLangShort(date){
    var d = new Date(date),
        month = d.getMonth() + 1,
        day = d.getDate();
        year = d.getFullYear();
        if (day.length < 2) 
            day = '0' + day;
    if (LANGUAGE == DE){
        return day + ". " + monthsDe[month-1];
        return day + ". " + monthsDe[month-1] + "\n" + year;
    }
    return monthsEn[month-1] + " " + day;
    return monthsEn[month-1] + " " + day + "\n" + year;
}

// Dare Formatter for Long Date View. e.g. Sep 10, 2020
function formatDateForLangLong(date){
    if (LANGUAGE === DE){
        var d = new Date(date),
            month = '' + (d.getMonth() + 1),
            day = '' + d.getDate(),
            year = d.getFullYear();
    
        if (month.length < 2) 
            month = '0' + month;
        if (day.length < 2) 
            day = '0' + day;
        return day + "." + month + "." + year;
    } else {
        return date.toISOString().slice(0,10);
    }
}

const monthsEn = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const monthsDe = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"];