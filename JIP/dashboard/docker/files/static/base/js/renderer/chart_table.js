class TableChart extends CovidChart {

    constructor(colorMeasure, sizeMeasure, language) {
        super(colorMeasure, sizeMeasure, language);

        this.chart = $('#tableChart');
        this.initDataDownloadCard();
    }

    initUI() {
        var nameKey =  `name_${this.language}`;

        // Function to calculate the last table row that shows the aggregated averages
        var avgCalc = function(values, data, calcParams){
            //values - array of column values
            //data - all table data
            //calcParams - params passed from the column definition object
            const cleanedValues = values.filter(function (a) { return a !== '-'});

            const sum = cleanedValues.reduce((a, b) => a + b, 0);
            const avg = (sum / cleanedValues.length) || 0;
        
            return parseFloat(avg.toFixed(DECIMALS));
        }

        var avgText = function(values, data, calcParams){
            return "Ø";
        }

        this.columns = [ { title: getI18n('city'), field: nameKey, bottomCalc: avgText } ];

        this.columns = [
            ...this.columns,
            ...MEASURES.map(measure => {
                return {
                    title: measure[nameKey],
                    field: measure.measure_id,
                    bottomCalc: avgCalc,
                    formatter: (cell, formatterParams, onRendered) => {
                        // Customize the cell here
                        let val = cell.getValue();
                        if (val == '-')
                            return val;

                        if (measure.measure_id === this.colorMeasure.measure_id && cell._cell && cell._cell.element) {
                            let delta = val - this.colorMin;
                            let range = this.colorMax - this.colorMin;
                            if (range == 0){
                                range = 1;
                                delta = 0.5;
                            }
                            var bg = '#000';
                            if (this.colorMiddle == null){
                                bg = blend_two_colors(this.colorLeft, this.colorRight, (delta/range).toFixed(2));
                            } else{
                                bg = blend_three_colors(this.colorLeft, this.colorMiddle, this.colorRight, (delta/range).toFixed(2));
                            }
                            let fg = invertColor(bg, true);

                            $(cell._cell.element).css('background-color', bg);
                            $(cell._cell.element).css('color', fg);
                        }
                        return val;
                    }
                }
            })
        ]

        // Options for Table. Refer to: http://tabulator.info/docs/4.9
        this.sortColumns();
        this.table = new Tabulator("#tableChart", {
            data: this.data,
            height:"100%",
            columns: this.columns,
            layout:"fitDataFill",
            resizableColumns:true,
            selectable:true,
            //responsiveLayout:"collapse"
            //responsiveLayoutCollapseStartOpen:false,
        });

        this.initLegends();
    }

    sortColumns() {
        // Sort columns. The metric selected for color encoding should be at position 1
        this.columns.sort((a, b) => {
            if (a.field === this.colorMeasure.measure_id) {
                return b.field === `name_${this.language}` ? 1 : -1;
            }
            else if (b.field === this.colorMeasure.measure_id) {
                return a.field === `name_${this.language}` ? -1 : 1;
            }
            return 0;
        })
    }

    initLegends() {
        this.colorLegend = new ColorLegend($('#colorLegendTable')[0]);

        this.colorLegend.setColors(this.colorLeft, this.colorMiddle, this.colorRight);
        this.colorLegend.setRange(this.colorMin, this.colorMax, this.colorMeasure[`name_${this.language}`]);
    }

    setColors(colorLeft, colorMiddle, colorRight) {
        this.colorLeft = colorLeft;
        this.colorRight = colorRight;
        this.colorMiddle = colorMiddle;

        if (this.colorLegend) this.colorLegend.setColors(colorLeft, colorMiddle, colorRight);
        this.refresh();
    }

    setColorMeasure(measure) {
        super.setColorMeasure(measure);

        let colorMeasureMeta = MEASURES.filter(ele => ele.measure_id === this.colorMeasure.measure_id)[0];
        this.colorMin = colorMeasureMeta.lower_bound;
        this.colorMax = colorMeasureMeta.upper_bound;

        if (this.columns) {
            this.sortColumns();
            this.table.setColumns(this.columns);
        }

        if (this.colorLegend) this.colorLegend.setRange(this.colorMin, this.colorMax, this.colorMeasure[`name_${this.language}`]);
    }

    resize() {
        this.table.replaceData(this.data);

        this.colorLegend.resize();
    }

    refresh() {
        if (!this.table) {
            this.initUI();
        }
        else {
            this.table.replaceData(this.data);
        }
    }

    initDataDownloadCard() {

        var lang = this.language;

        function download(file, text) {              
            //creating an invisible element that triggers the download
            var element = document.createElement('a');
            element.setAttribute('href', 'data:text/plain;charset=utf-8, ' + encodeURIComponent(text));
            element.setAttribute('download', file);            
          
            document.body.appendChild(element);
            element.click();          
            document.body.removeChild(element);
        }

        // predefined date ranges of the download menu
        var ranges = {};
        ranges[getI18n('date_range_today')] = [moment(), moment()],
        ranges[getI18n('date_range_yesterday')] = [moment().subtract(1, 'days'), moment().subtract(1, 'days')]
        ranges[getI18n('date_range_7_days')] = [moment().subtract(6, 'days'), moment()]
        ranges[getI18n('date_range_30_day')] = [moment().subtract(29, 'days'), moment()]
        ranges[getI18n('date_range_this_month')] = [moment().startOf('month'), moment().endOf('month')]
        ranges[getI18n('date_range_last_month')] = [moment().subtract(1, 'month').startOf('month'), moment().subtract(1, 'month').endOf('month')]

        // Localization for the download menu
        var locale = {
            "format": "YYYY-MM-DD",
            "separator": " - ",
            "applyLabel": "Apply",
            "cancelLabel": "Cancel",
            "fromLabel": "From",
            "toLabel": "To",
            "customRangeLabel": "Custom Range",
            "weekLabel": "W",
            "daysOfWeek": ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"],
            "monthNames": [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December"
            ],
            "firstDay": 1
        };

        if (lang == 'de') {
            var locale = {
                "format": "DD.MM.YYYY",
                "separator": " - ",
                "applyLabel": "Speichern",
                "cancelLabel": "Abbrechen",
                "fromLabel": "Von",
                "toLabel": "Bis",
                "customRangeLabel": "Eigener Zeitraum",
                "weekLabel": "W",
                "daysOfWeek": ["So", "Mo", "Di", "Mi", "Do", "Fr", "Sa"],
                "monthNames": [
                    "Januar",
                    "Februar",
                    "März",
                    "April",
                    "Mai",
                    "Juni",
                    "Juli",
                    "August",
                    "September",
                    "Oktober",
                    "November",
                    "Dezember"
                ],
                "firstDay": 1
            };
        }

        // Initialize Date Range Picker. Refer to: https://www.daterangepicker.com/
        $('input[name="daterange"]').daterangepicker({
            startDate: moment().subtract(29, 'days'),
            endDate: moment(),
            locale: locale,
            ranges: ranges
        });

        $('#dataDownloadButton').click(function () {
            $('#dataDownloadButton').text(getI18n('download_loading'));
            $('#dataDownloadButton').append(`<span id="dataDownloadSpinner" class="ml-3 mb-1 spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>`);
            $('#dataDownloadButton').prop('disabled', true);
            // Build API URL to generate download
            var startDate = $('#dataDownloadRange').data('daterangepicker').startDate.format('YYYY-MM-DD');
            var endDate = $('#dataDownloadRange').data('daterangepicker').endDate.format('YYYY-MM-DD');
            var url = `${ SITE_URL }/api/export?`;
            url += `date__gte=${startDate}`
            url += `&date__lte=${endDate}`
            
            if ($('#filter-options').length > 0) {
                var op= $("#filter-options")[0];
                let selected_locations = [...op.options].filter(option => option.selected).map(option => option.value).join(',');
                url += `&location__in=${selected_locations}`;
            }

            // Get data from url
            $.ajax({
                type: 'GET',
                url: url,
                headers:{
                    "X-CSRFToken": getCookie('csrftoken')
                },
                success: function (result) {
                    download("data.csv", result);
                    $('#dataDownloadSpinner').remove();
                    $('#dataDownloadButton').text(getI18n('download_btn'));
                    $('#dataDownloadButton').prop('disabled', false);
                }, 
                error: function (error) {
                    $('#dataDownloadSpinner').remove();
                    $('#dataDownloadButton').text(getI18n('download_error'));
                    $('#dataDownloadButton').prop('disabled', false);
                }
            })
        });

    }

}

