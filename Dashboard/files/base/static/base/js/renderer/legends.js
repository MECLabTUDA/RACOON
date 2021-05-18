class ColorLegend {

    constructor(element) {
        this.element = element;
        this.min = 0;
        this.max = 10;
        this.colorLeft = "#F5F5F5";
        this.colorRight = "#EE7218";

        this.$header = $('<span style="font-size: smaller;">Sample Title</span>');
        this.$content = $(`
            <div class="mt-4" style="display:flex; flex-direction:row">
                <span name="min">0</span>
                <div name="gradient" class="ml-2 mr-2" style="flex:1; background-image: linear-gradient(to right, ${this.colorLeft}, ${this.colorRight});"></div>
                <span name="max">10</span>
            </div>
        `);
		
		this.$previewContent = $(`
        <div class="mt-3 mb-4" style="display:flex; flex-direction:row">
            <div name="gradient" class="ml-2 mr-2" style="height:30px; flex:1; background-image: linear-gradient(to right, ${this.colorLeft}, ${this.colorRight});"></div>
        </div>
        `);

        this.renderLegend();
    }

    renderLegend() {
        $(this.element).empty();
        $(this.element).append(this.$header);
        $(this.element).append(this.$content);
    }

    resize() {
        // Do Noting
    }

    setRange(min, max, title) {
        this.min = min;
        this.max = max;

        this.$header.text(title);
        this.$content.find("[name=min]").text(Math.floor(this.min).toFixed(0));
        this.$content.find("[name=max]").text(Math.ceil(this.max).toFixed(0));
    }

    setColors(colorLeft, colorMiddle, colorRight) {
        this.colorLeft = colorLeft;
        this.colorRight = colorRight;
        this.colorMiddle = colorMiddle;
        if (colorMiddle == null){
            this.$content.find("[name=gradient]").css('background-image', `linear-gradient(to right, ${this.colorLeft}, ${this.colorRight})`);
            this.$previewContent.find("[name=gradient]").css('background-image', `linear-gradient(to right, ${this.colorLeft}, ${this.colorRight})`);
        } else {
            this.$content.find("[name=gradient]").css('background-image', `linear-gradient(to right, ${this.colorLeft}, ${this.colorMiddle}, ${this.colorRight})`);
            this.$previewContent.find("[name=gradient]").css('background-image', `linear-gradient(to right, ${this.colorLeft}, ${this.colorMiddle}, ${this.colorRight})`);
        }
    }

	setPreviewColors(colorLeft, colorMiddle, colorRight) {
        if (colorMiddle == null) {
            this.$previewContent.find("[name=gradient]").css('background-image', `linear-gradient(to right, ${colorLeft}, ${colorRight})`);
        } else {
            this.$previewContent.find("[name=gradient]").css('background-image', `linear-gradient(to right, ${colorLeft}, ${colorMiddle}, ${colorRight})`);
        }
    }

	setPreviewElement(element) {        
        $(element).append(this.$previewContent);
    }
}


class SizeLegend {

    constructor(element) {
        this.element = element;
        this.min = 0;
        this.max = 100;

        this.$header = $('<span style="font-size: smaller;">Sample Title</span>');
        this.$content = $(`<div style="height: 5rem;"></div>`);

        // Apache Echarts definition of Legend Design
        this.option = {
            backgroundColor: THEME === 'dark' ? '#424242' : '#FFF',
            xAxis: {
                show: false
            },
            yAxis: {
                show: false,
                type: 'category'
            },
            visualMap: {
                show: false,
                hoverLink: false,
                type: 'continuous',
                dimension: 0,
                seriesIndex: 0,
                min: this.min,
                max: this.max,
                text: [Math.ceil(this.max).toFixed(0), Math.floor(this.min).toFixed(0)],
                inRange: {
                    symbolSize: [15, 45]
                }
            },
            series: {
                color: 'gray',
                label: {
                    show: true,
                    formatter: function(d) {
                        return d.data[2];
                    },
                    position: 'bottom'
                },
                data: [
                    // data[0] = x coordinate (equally distributed), data[1] = y coordinate (always on same horizontal line), data[2] = actual label to display 
                    [0, 0, 0],
                    [25, 0, 2.5],
                    [50, 0, 5],
                    [75, 0, 7.5],
                    [100, 0, 10]
                ],
                type: 'scatter',
                itemStyle: {
                    color: THEME === 'dark' ? '#ECECEC' : '#7c7e86',
                }
            }
        };

        this.renderLegend();
    }

    renderLegend() {
        $(this.element).empty();
        $(this.element).append(this.$header);
        $(this.element).append(this.$content);

        this.chart = echarts.init(this.$content[0], THEME);
        this.chart.setOption(this.option);

        setTimeout(() => this.resize(), 10);
    }

    resize() {
        this.chart.resize();
    }

    setRange(min, max, title) {
        this.$header.text(title);

        let range = max - min;
        let data = [
            [0, 0, max <= 10 ? min.toFixed(2) : Math.round(min.toFixed(2))],
            [25, 0, max <= 10 ? (min + (range) * 0.25).toFixed(2) : Math.round((min + (range) * 0.25).toFixed(2))],
            [50, 0, max <= 10 ? (min + (range) * 0.50).toFixed(2) : Math.round((min + (range) * 0.50).toFixed(2))],
            [75, 0, max <= 10 ? (min + (range) * 0.75).toFixed(2) : Math.round((min + (range) * 0.75).toFixed(2))],
            [100, 0, max <= 10 ? Math.ceil(max).toFixed(2) : Math.round(Math.ceil(max).toFixed(2))],
        ]

        this.chart.setOption({
            series: {
                data: data
            }
        });
    }
}
