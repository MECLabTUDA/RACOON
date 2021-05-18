class AggregationChart extends CovidChart {

    constructor(colorMeasure, sizeMeasure, language) {
        super(colorMeasure, sizeMeasure, language);
        
        this.aggregationChart = echarts.init(document.getElementById('aggregationChart'), THEME);
    }

    async refresh() {
        var lookback_days = 30
        // Generate 30 Days History Array
        var days = [];
        var days_display = []
        var currentDay = dayjs(this.date);
        for (let step = 0; step < lookback_days; step++) {
            days.push(currentDay.format('YYYY-MM-DD'));
            days_display.push( this.language === "en" ? currentDay.format('YYYY-MM-DD') : currentDay.format('DD.MM.YYYY') );
            currentDay = currentDay.subtract(1, 'day');
        }
        days = days.reverse();
        days_display = days_display.reverse();

        // Get 30 Days History Data from API
        let url = `${ API_URL }/avg/?date__lte=${ dayjs(this.date).format('YYYY-MM-DD') }&date__gte=${ dayjs(this.date).subtract(lookback_days-1, 'day').format('YYYY-MM-DD') }`;
        if ($('#filter-options').length > 0) {
            var op = $("#filter-options")[0];
            let selected_locations = [...op.options].filter(option => option.selected).map(option => option.value).join(',');
            if (selected_locations != ''){
                url += `&location__in=${selected_locations}`;                
            }
        }
        const res = await axios.get(url);

        // Generate Measure Name Array
        var measure_ids = Object.keys(res['data']);  
        var measureNames = measure_ids.map(mid => {
            return MEASURES.filter(ele => ele.measure_id === mid)[0][`name_${this.language}`];
        })

        // Generate Plot Data Matrix
        var plotData = [];
        var measureMaxima = [];
        var measureMinima = [];
        var data = res.data;
        var x = 0;
        for (var measure in data) {
            var max = Number.MIN_VALUE;
            var min = Number.MAX_VALUE;
            var y = 0;
            days.forEach(function(day){
                if (data[measure].hasOwnProperty(day)){
                    plotData.push([x, y, data[measure][day].toFixed(DECIMALS_AGG)]);
                    if (data[measure][day] > max){
                        max = data[measure][day];
                    }
                    if (data[measure][day] < min){
                        min = data[measure][day];
                    }
                } else {
                    plotData.push([x, y, '-']);
                }
                y = y + 1;
            })
            measureMaxima[measureNames[x]] = max;
            measureMinima[measureNames[x]] = min;
            x = x + 1;    
        }

        // obtain the upper and lower bound for all metrics to normalize the data in the plot for color encoding
        let globalMeasureMinima = MEASURES.reduce((a,x) => ({...a, [x[`name_${this.language}`]]: x.lower_bound}), {})
        let globalMeasureMaxima = MEASURES.reduce((a,x) => ({...a, [x[`name_${this.language}`]]: x.upper_bound}), {})

        var width = (window.innerWidth > 0) ? window.innerWidth : screen.width;
        
        // Apache Echarts plot definition
        var option = {
            backgroundColor: THEME === 'dark' ? '#424242' : '#FFFFFF',
            tooltip: {
                position: 'top',
                trigger:"axis",
                //triggerOn: "none",
                formatter: function (params) {
                    var dataIndex = params[0].dataIndex;
                    var seriesIndex = params[0].seriesIndex;
                    var date = params[0].axisValue;
                    var tooltip = `<u>${date}</u><br/><table style="table-layout: fixed;">`
                    for (const [idx, mid] of measure_ids.entries()){
                        var measureName = measureNames[idx];
                        var value = data[mid][days[dataIndex]];
                        if (value == null){
                            value = '-';
                        } else {
                            value = data[mid][days[dataIndex]].toFixed(DECIMALS_AGG)
                        }
                                                
                        if (idx == seriesIndex){
                            tooltip +=  `<tr><td style="word-wrap: break-word; white-space: normal; max-width: ${width < 974 ? 130 : 250}px; padding:0 5px 0 0;">${params[0].marker} <b>${measureName}</b>:</td><td><b>${value}</b></td></tr>`
                        } else {
                            tooltip +=  `<tr><td style="word-wrap: break-word; white-space: normal; max-width: ${width < 974 ? 120 : 250}px; padding:0 5px 0 0;">• &nbsp;${measureName}:</td><td>${value}</td></tr>`
                        }
                    }
                    tooltip += `</table>`;
                    return tooltip
                },
                textStyle: {
                    fontSize: (width < 974) ? 12 : 14                    
                }
            },
            title: [],
            grid: [],
            visualMap: [],
            xAxis: [],
            yAxis: [],
            series: []
        };

        var colorRange = [this.colorLeft, this.colorMiddle, this.colorRight]
        if (this.colorMiddle == null){
            var colorRange = [this.colorLeft, this.colorRight]
        }

        echarts.util.each(measureNames, function (measureName, idx) {
            option.xAxis.push({
                gridIndex: idx,
                type: 'category',
                nameLocation: "middle",
                name: measureName,
                position: "top",
                data: days_display,
                axisTick: {
                    show: false
                },
                axisLabel: {
                    show: false
                },
                axisLine: {
                    show: false
                },
                nameGap: 2,
            });
            option.yAxis.push({
                gridIndex: idx,
                type: 'category',
                show: false,
                data: [measureName],
                name: measureName,
                splitArea: {
                    show: false
                }
            });
            option.grid.push({
                type: 'category',
                boundaryGap: false,
                data: days,
                top: ((idx==0) ? (idx * 95 / measureNames.length) + 7 + "%" : (idx * 90 / measureNames.length) + 14 + "%"),
                height: 95 / (measureNames.length*2.5) + '%',
                axisLabel: {
                    interval: 2
                }
            });
            option.visualMap.push({
                seriesIndex: idx,
                // globalMeasureMin/Max is used for a global color encoding over all data.
                // if the encoding should only be done for the selected range, use the following instead:
                //min: measureMinima[measureName],
                //max: measureMaxima[measureName],
                min: globalMeasureMinima[measureName],
                max: globalMeasureMaxima[measureName],
                calculable: true,
                orient: 'horizontal',
                left: 'center',
                show: false,
                top: idx * 100 / measureNames.length + 2 + "%",
                inRange: {
                    color: colorRange
                }
            });
            option.series.push({
                name: measureName,
                xAxisIndex: idx,
                yAxisIndex: idx,
                type: 'heatmap',
                data: [],
                label: {
                    show: false
                },
                emphasis: {
                    itemStyle: {
                        shadowBlur: 10,
                        shadowColor: 'rgba(0, 0, 0, 0.8)',
                        borderColor: "#333",
                        borderWidth: 1
                    }
                }
            });
        });

        option.xAxis.push({
            gridIndex: 0,
            type: 'category',
            data: days_display,
            splitArea: {
                show: false
            },
            axisTick: {
                alignWithLabel: true,
                interval: 'auto',
            },
            axisLabel: {
                showMinLabel: true,
                showMaxLabel: true,
            },
            boundaryGap: true,
        });

        // Add a chart to the plot for each available measure
        echarts.util.each(plotData, function (dataItem) {
            option.series[dataItem[0]].data.push([dataItem[1], 0, dataItem[2]]);
        });

        this.option = option;
        this.aggregationChart.setOption(option, true);
        // define the custom tooltip that shows the data for all metrics in a vertical slice on hover
        var ac = this.aggregationChart;
        this.aggregationChart.on('mouseover', function (params) {
            ac.dispatchAction({                                
                type: 'highlight',
                seriesIndex: [...Array(measureNames.length).keys()],
                dataIndex: params.dataIndex,
            })
        });
        this.aggregationChart.on('mouseout', function (params) {
            ac.dispatchAction({                                
                type: 'downplay',
                seriesIndex: [...Array(measureNames.length).keys()],
                dataIndex: params.dataIndex,
            })
        });
    }


    setColors(colorLeft, colorMiddle, colorRight) {
        this.colorLeft = colorLeft;
        this.colorRight = colorRight;
        this.colorMiddle = colorMiddle;
        
        for (let idx = 0; idx < this.option.visualMap.length; idx++) {
            if (colorMiddle == null){
                this.option.visualMap[idx].inRange.color = [colorLeft, colorRight];
            } else {
                this.option.visualMap[idx].inRange.color = [colorLeft, colorMiddle, colorRight];
            }
        }

        this.aggregationChart.setOption(this.option);
        this.resize()
    }
 

    resize() {
        let w = $('#aggregationChart').width();
        if (w === this.aggregationChartWidth) return;

        this.aggregationChartWidth = w;

        var width = (window.innerWidth > 0) ? window.innerWidth : screen.width;
        this.option.tooltip.textStyle.fontSize = (width < 974) ? 12 : 14;
        this.aggregationChart.setOption(this.option);

        this.aggregationChart.resize();
    }

    setDateSlider(dateSlider){
        this.dateSlider = dateSlider;
    }

}
