class ChartManager {

    constructor() {

        this.language = LANGUAGE;

        this.selectedDate = today();

        this.activeChart = MAP_CHART;
        this.data = [];

        this.dataLoadTimer = null;

        this.mainMeasures = [];
        this.nonMainMeasures = [];
        this.colorMeasure = null;
        this.sizeMeasure = null;
        this.charts = [];

        this.loadMetaData(() => {
            this.charts = [
                new MapChart(this.colorMeasure, this.sizeMeasure, this.language),
                new TableChart(this.colorMeasure, this.sizeMeasure, this.language),
                new AggregationChart(this.colorMeasure, this.sizeMeasure, this.language)
            ];
            this.loadData();
            this.initUI();
            this.initFilterOptions();
        });
        
    }

    async loadMetaData(callback) {
        const locations = await axios.get(`${API_URL}/locations`);
        const measures = await axios.get(`${API_URL}/measures`);
        const meta = await axios.get(`${API_URL}/meta`);

        locations.data.forEach(location => {
            LOCATIONS[location.location_id] = location;
        });

        MEASURES.push(...measures.data);
        this.mainMeasures = MEASURES.filter(measure => measure["is_main"]);
        this.nonMainMeasures = MEASURES.filter(measure => !measure["is_main"]);
        if (MEASURES.filter(measure => measure["is_color_default"]).length > 0){
            this.colorMeasure = MEASURES.filter(measure => measure["is_color_default"])[0];
        } else {
            this.colorMeasure = this.mainMeasures[0];
        }
        if (MEASURES.filter(measure => measure["is_size_default"]).length > 0){
            this.sizeMeasure = MEASURES.filter(measure => measure["is_size_default"])[0];
        } else {
            this.sizeMeasure = this.colorMeasure;
        }

        MEASURES.forEach(measure => {
            let measureMeta = meta.data.filter(ele => ele.measure_id === measure.measure_id)[0];
            measure.lower_bound = measure.is_open_ended ? measureMeta.min_value : measure.lower_bound;
            measure.upper_bound = measure.is_open_ended ? measureMeta.max_value : measure.upper_bound;
        });

        this.initColorAndSizeOptions();

        callback();
    }

    loadData() {
        if (this.dataLoadTimer) {
            clearTimeout(this.dataLoadTimer);
            this.dataLoadTimer = null;
        }
        // This allows user some time to finalize the value
        this.dataLoadTimer = setTimeout(async () => {
            let url = `${ API_URL }/data/?date=${ dayjs(this.selectedDate).format('YYYY-MM-DD') }`;
            if ($('#filter-options').length > 0) {
                var op= $("#filter-options")[0];
                let selected_locations = [...op.options].filter(option => option.selected).map(option => option.value).join(',');
                url += `&location__in=${selected_locations}`;
            }
            const res = await axios.get(url);
            // create a dict from all nested measures and merge them with the corresponding location
            this.data = res.data.map(entry => { 
                let present_measures = entry.measureData.length > 0 ? {...Object.assign(...entry.measureData.map((me) => ({[me.measure]: me.value})))} : {};
                return {
                    ...Object.assign(
                        ...MEASURES.map( (measure) => (
                            {[measure.measure_id] : ($.inArray(measure.measure_id, Object.keys(present_measures)) != -1) ? parseFloat(present_measures[measure.measure_id].toFixed(DECIMALS)) : '-'}
                        )) 
                    ), 
                    ...{['date']: entry.date}, 
                    ...LOCATIONS[entry.location],
                    ...{['plotData']: entry.plotData}
                } 
            });

            this.handleDataChange();
        }, 150);
    }

    initUI() {
        this.sliderContext = {
            selectedDate: new Date(this.selectedDate.getTime()),
            startDate: new Date(this.selectedDate.getTime()),
            endDate: new Date(this.selectedDate.getTime())
        }

        this.rangeSlider1 = new DateRangeSlider(this, this.sliderContext, 'map', ['table', 'aggregation'], this.language);
        this.rangeSlider2 = new DateRangeSlider(this, this.sliderContext, 'table', ['map', 'aggregation'], this.language);
        this.rangeSlider3 = new DateRangeSlider(this, this.sliderContext, 'aggregation', ['map', 'table'], this.language);

        this.charts[MAP_CHART].setDateSlider(this.rangeSlider1);

        $('#map-tab').on('click', e => setTimeout(() => this.setActiveChart(MAP_CHART), 200));
        $('#data-tab').on('click', e => setTimeout(() => this.setActiveChart(TABLE_CHART), 200));
        $('#aggregation-tab').on('click', e => setTimeout(() => this.setActiveChart(AGGREGATION_CHART), 200));

        $('#languageSwitch').on('change', e => {
            let url = `${SITE_URL}/${ THEME === 'light' ? '' : 'dark/'}?lang=${LANGUAGE === EN ? DE : EN}`
            window.location.replace(url);
        });
    }

    refreshDateSliders() {
        this.rangeSlider1.refresh();
        this.rangeSlider2.refresh();
        this.rangeSlider3.refresh();
    }

    initFilterOptions(){
        var parent = $("#filterModalBody");

        //Create and append select list
        var selectList = document.createElement("select");
        selectList.id = "filter-options";
        selectList.multiple = "multiple";
        parent.append(selectList);

        //Create and append the options
        for (const [location_id, location] of Object.entries(LOCATIONS)) {
            var option = document.createElement("option");
            option.value = location_id;
            option.text = location[`name_${this.language}`];
            //option.selected = true;
            selectList.appendChild(option);
        }

        multi(selectList, {
            "enable_search": true,
            "search_placeholder": getI18n('filter_search'),
            "limit": -1,
            "limit_reached": function () {},
            "hide_empty_groups": false,
            'non_selected_header': getI18n('filter_available'),
            'selected_header': getI18n('filter_selected')
        });
    }

    initColorAndSizeOptions() {
        var colorOptions = $("#color-options");
        var sizeOptions = $("#size-options");

        colorOptions.empty();
        sizeOptions.empty();

        this.mainMeasures.forEach((measure, i) => {
            var radio = $(`
                <div class="row align-items-center ml-0 mr-0">
                    <span class="col-11">
                        <label class="form-check-label" for="c_${measure.measure_id}">
                            <input type="radio" class="form-check-input" id="c_${measure.measure_id}" name="colorRadio" value="${measure.measure_id}" ${measure.measure_id === this.colorMeasure.measure_id ? 'checked' : '' }>${measure[`name_${this.language}`]}
                        </label>
                    </span>
                    <span>
                        <a href="#" class="fa fa-info-circle" data-placement="top" data-trigger="hover focus" data-toggle="popover" data-original-title="${measure[`name_${this.language}`]}" data-content="${measure[`description_${this.language}`]}"></a>
                    </span>
                </div>
            `);

            colorOptions.append(radio);
            radio.find('input').on('click', e => this.setColorMeasure(measure));
        });

        this.mainMeasures.forEach((measure, i) => {
            var radio = $(`
                <div class="row align-items-center ml-0 mr-0">
                    <span class="col-11">
                        <label class="form-check-label" for="s_${measure.measure_id}">
                            <input type="radio" class="form-check-input" id="s_${measure.measure_id}" name="sizeRadio" value="${measure.measure_id}" ${measure.measure_id === this.sizeMeasure.measure_id ? 'checked' : '' }>${measure[`name_${this.language}`]}
                        </label>
                    </span>
                    <span>
                        <a href="#" class="fa fa-info-circle" data-placement="top" data-trigger="hover focus" data-toggle="popover" data-original-title="${measure[`name_${this.language}`]}" data-content="${measure[`description_${this.language}`]}"></a>
                    </span>
                </div>
            `);

            radio.find('input').on('click', e => this.setSizeMeasure(measure));
            sizeOptions.append(radio);
        });

        $('[data-toggle="popover"]').popover();

    }

	setPreviewColors(colorLeft, colorMiddle, colorRight){
        this.charts[MAP_CHART].setPreviewColors(colorLeft, colorMiddle, colorRight);
	}

    setColors(colorLeft, colorMiddle, colorRight) {
        this.colorLeft = colorLeft;
        this.colorRight = colorRight;
        this.colorMiddle = colorMiddle;
        this.charts[MAP_CHART].setColors(colorLeft, colorMiddle, colorRight);
        this.charts[TABLE_CHART].setColors(colorLeft, colorMiddle, colorRight);
        this.charts[AGGREGATION_CHART].setColors(colorLeft, colorMiddle, colorRight);
    }

    setColorMeasure(measure) {
        this.colorMeasure = measure;
        this.charts[MAP_CHART].setColorMeasure(measure);
        this.charts[TABLE_CHART].setColorMeasure(measure);

        this.handleDataChange();
    }

    setSizeMeasure(measure) {	
        this.sizeMeasure = measure;
        this.charts[MAP_CHART].setSizeMeasure(measure);
        this.charts[TABLE_CHART].setSizeMeasure(measure);

        this.handleDataChange();
    }

    setActiveChart(activeChart) {
        this.activeChart = activeChart;
        this.charts[this.activeChart].resize();
    }

    setDate(currentValue) {
        if (currentValue == this.selectedDate.getTime()) return; 
        this.selectedDate = new Date(currentValue);
        this.loadData();
    }

    handleDataChange() {
        this.charts.forEach((chart, i) => {
            chart.setData(this.selectedDate, this.data);
        });

        this.charts[MAP_CHART].refresh();
        this.charts[TABLE_CHART].refresh();
        this.charts[AGGREGATION_CHART].refresh();
    }

    handleWindowResize() {
        if (this.charts.length > 0)
            this.charts[this.activeChart].resize();
        if (this.rangeSlider1) this.rangeSlider1.resize();
        if (this.rangeSlider2) this.rangeSlider2.resize();
        if (this.rangeSlider3) this.rangeSlider3.resize();
    }

    reloadAll(){
        // Reload accessible measures
        MEASURES.splice(0,MEASURES.length);
        this.loadData();
        this.loadMetaData(() => {
            //Update Selected Settings
            //this.setColorMeasure(this.colorMeasure);
            //this.setSizeMeasure(this.sizeMeasure);

            this.charts[TABLE_CHART].initUI();
            this.charts.forEach((chart, i) => {
                chart.setData(this.selectedDate, this.data);
                chart.refresh();
                chart.setColorMeasure(this.colorMeasure);
                chart.setSizeMeasure(this.sizeMeasure);
            });
        });

    }
}
