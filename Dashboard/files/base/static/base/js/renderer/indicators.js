class IndicatorChart {

    constructor(element) {
        this.element = element;
        this.min = 0;
        this.max = 100;

        this.$header = $('<span style="font-size: smaller;">Sample Title</span>');
        this.$content = $(`<div style="height: 8rem; display: flex; justify-content: center;"></div>`);

        // Apache Echart definition of the the look of the indicators
        this.option = {
            backgroundColor: THEME === 'dark' ? '#424242' : '#FFFFFF',  
            series: {                
                    type: 'gauge',
                    center: ["50%", "70%"],
                    startAngle: 190,
                    endAngle: -10,
                    splitNumber: 6,
                    itemStyle: {
                        color: '#FFAB91',
                    },
                    progress: {
                        show: true,
                        width: 20
                    },
        
                    pointer: {
                        show: false,
                    },
                    axisLine: {
                        lineStyle: {
                            width: 20,
                            color: [[1, THEME === 'dark' ? '#5e5f61' : '#f0f2f5']],
                        }
                    },
                    axisTick: {
                        distance: -27,
                        splitNumber: 5,
                        lineStyle: {
                            width: 1,
                            color: THEME === 'dark' ? '#cccccc' : '#999',
                        }
                    },
                    splitLine: {
                        distance: -32,
                        length: 10,
                        lineStyle: {
                            width: 2,
                            color: THEME === 'dark' ? '#cccccc' : '#999',
                        }
                    },
                    axisLabel: {
                        distance: -12,
                        color: THEME === 'dark' ? '#cccccc' : '#999',
                        fontSize: 12,
                        padding: [2, -8, 0, -8],
                        formatter: (function(d){
                            var num = ((d > 100) ? Math.round(d) : parseFloat(d.toFixed(1))) ;
                            return num;
                        }),
                    },
                    anchor: {
                        show: false
                    },
                    title: {
                        offsetCenter: [0, '65%'],
                        fontSize: 13,
                        show: true,
                        color: THEME === 'dark' ? '#dbdbdb' : '#7d7d7d',
                    },
                    detail: {
                        valueAnimation: true,
                        width: '60%',
                        lineHeight: 40,
                        height: '15%',
                        borderRadius: 8,
                        offsetCenter: [0, '-15%'],
                        fontSize: 12,
                        fontWeight: 'bolder',
                        formatter: (function(d){
                            var num = ((d > 100) ? parseFloat(d.toFixed(1)) : d);
                            return num;
                        }),
                        color: THEME === 'dark' ? '#cccccc' : '#999',
                        //color: 'auto',
                    },
                    data: [{
                        value: 0,
                        name: 'Titel'
                    }]
                }
        };
        this.renderIndicator();
    }

    renderIndicator() {
        $(this.element).empty();
        $(this.element).append(this.$content);

        this.chart = echarts.init(this.$content[0], THEME);
        this.chart.setOption(this.option);

        setTimeout(() => this.resize(), 10);
    }

    resize() {
        this.chart.resize();
    }

    setMeasure(min, max, title) {
        this.min = min;
        this.max = max;
        
        this.option.series.min = min;
        this.option.series.max = max;
        this.option.series.data[0].name = 'Ø ' + title;
        this.chart.setOption(this.option);
    }

    setData(data, displayDimension){
        let sum = 0;
        let count = 0;

        data.map(entry => { 
            sum += entry[displayDimension] === '-' ? 0 : entry[displayDimension];
            count += entry[displayDimension] === '-' ? 0 : 1;
        });

        let avg = count == 0 ? 0 : sum / count;
        var avgFixed = avg.toFixed(2);

        this.option.series.data[0].value = avgFixed;
        this.chart.setOption(this.option)
        
        this.updateDisplayColors();
    }

    setColors(colorLeft, colorMiddle, colorRight) {
        this.colorLeft = colorLeft;
        this.colorRight = colorRight;
        this.colorMiddle = colorMiddle;
        this.updateDisplayColors();
    }

    updateDisplayColors(){
        var value = this.option.series.data[0].value;
        let delta = value - this.min;
        let range = this.max - this.min;
        if (range == 0){
            range = 1;
            delta = 0.5;
        }

        var color = '#000';
        if (this.colorMiddle == null){
            color = blend_two_colors(this.colorLeft, this.colorRight, (delta/range).toFixed(2));
        } else{
            color = blend_three_colors(this.colorLeft, this.colorMiddle, this.colorRight, (delta/range).toFixed(2));
        }
        this.option.series.itemStyle.color = color;
        this.chart.setOption(this.option)        
    }
}
