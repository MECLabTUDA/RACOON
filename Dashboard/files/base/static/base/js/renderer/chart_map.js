class MapChart extends CovidChart {

    constructor(colorMeasure, sizeMeasure, language) {
        super(colorMeasure, sizeMeasure, language);

        this.chartWidth = 0;
        this.scaleLimit = {
            min: 1,
            max: 10
        }

        this.currentModalLocation = "";
        
        this.dotMinSize = 15;
        this.dotMaxSize = 45;
        //this.date = new Date();

        this.isMobile = window.matchMedia("only screen and (max-width: 760px)").matches;       
        if (this.isMobile) {
            this.chart = echarts.init(document.getElementById('mapChart'), THEME, {renderer: 'svg'});
        } else {
            this.chart = echarts.init(document.getElementById('mapChart'), THEME, {renderer: 'canvas'});
        }
        
        this.chart.showLoading();
        
        this.stateBorderData = [];
        this.geoRegionData = [];
        this.createOptions();        

        this.initIndicators();

        this.loadGeoJson();

        //this.initPlots();
    }

    createZoomButtons() {
        return {
            elements: [
                {
                    type: 'rect',
                    top: 10,
                    right: 5,
                    shape: {
                        width: 21,
                        height: 21
                    },
                    style: {
                        fill: '#FFF',
                        shadowBlur: 1,
                        stroke: '#5a5c69',
                        shadowColor: '#c2c2c2',
                    },
                    z: 3,
                    onclick: this.buttonZoomIn.bind(this)
                },
                {
                    type: 'text',
                    top: 10,
                    right: 5,
                    style: {                        
                        text: '＋',
                        fontSize: 22,
                    },
                    z: 3,
                    onclick: this.buttonZoomIn.bind(this)
                },
                {
                    type: 'rect',
                    top: 35,
                    right: 5,
                    shape: {
                        width: 21,
                        height: 21
                    },
                    style: {
                        fill: '#FFF',
                        shadowBlur: 1,
                        stroke: '#5a5c69',
                        shadowColor: '#c2c2c2',
                    },
                    z: 3,
                    onclick: this.buttonZoomOut.bind(this)
                },
                {
                    type: 'text',
                    top: 37,
                    right: 11,
                    style: {                        
                        text: '–',
                        fontSize: 18,
                        fontWeight: 'bold',
                    },
                    z: 3,
                    onclick: this.buttonZoomOut.bind(this)
                },
                {
                    type: 'rect',
                    top: 60,
                    right: 5,
                    shape: {
                        width: 21,
                        height: 21
                    },
                    style: {
                        fill: '#FFF',
                        shadowBlur: 1,
                        stroke: '#5a5c69',
                        shadowColor: '#c2c2c2',
                        font: '1.0em sans-serif'
                    },
                    z: 3,
                    onclick: this.zoomReset.bind(this)
                },
                {
                    type: 'text',
                    top: 64,
                    right: 12,
                    style: {                        
                        text: '0',
                        fontSize: 15,
                        fontFamily: 'arial',
                        fontWeight: 'bold',
                    },
                    z: 3,
                    onclick: this.zoomReset.bind(this)
                },
            ]
        }
    }

    createOptions() {	
        this.option = {
            backgroundColor: THEME === 'dark' ? '#424242' : '#FFFFFF',
            graphic: this.createZoomButtons(),
            geo: {
                map: 'Germany',
                roam: true,
                scaleLimit: {
                    min: this.scaleLimit.min,
                    max: this.scaleLimit.max,
                },
                label: {
                    show: false,
                    emphasis: {
                        show: false
                    }
                },
                itemStyle: {
                    borderColor: '#BBB9BA',
                    areaColor: '#EEEEEE',
                    borderWidth: 1,
                    emphasis: {
                        borderColor: '#BBB9BA',
                        areaColor: '#EEEEEE',
                        borderWidth: 1,
                    }
                },
                zoom: 1,
                regions: [],
            },
            visualMap: [
                {
                    type: 'continuous',
                    dimension: 5,
                    seriesIndex: 0,
                    inRange: {
                        symbolSize: [this.dotMinSize, this.dotMaxSize]
                    },
                    show: false,
                    hoverLink: false,
                },
                {
                    type: 'continuous',
                    seriesIndex: 0,
                    dimension: 6,
                    inRange: {
                        color: [this.colorLeft, this.colorRight]
                    },
                    show: false,
                    hoverLink: false,
                }
            ],
            tooltip: {
                show: true,
				formatter: function(params) {
					// remove unwanted suffix from dimension names. If dimNameA = dimNameB, dimB will be renamed to dimB-1. The next two lines prevent this
					let dimA = params.dimensionNames[3].replace(/-[0-9]$/, "");
					let dimB = params.dimensionNames[4].replace(/-[0-9]$/, "");
					let tooltip = `<p>${params.seriesName}: ${params.value[0]}</p>`;
					tooltip += `<table>`;
					tooltip += `<tr><td style="padding:0 5px 0 0;">• &nbsp; ${dimA}:</td><td>${params.value[3] === '-' ? '-' : Math.round((params.value[3] + Number.EPSILON) * 100) / 100}</td></tr>`;
					tooltip += `<tr><td style="padding:0 5px 0 0;">${params.marker} ${dimB}:</td><td>${params.value[4] === '-' ? '-' : Math.round((params.value[4] + Number.EPSILON) * 100) / 100}</td></tr>`;
					tooltip += `</table>`;
					return tooltip;	
				},
            },
            series: [
                {
                    name: getI18n('location'),
                    type: "scatter",
                    coordinateSystem: 'geo',
                    connectNulls:false,
                    itemStyle: {
                        borderColor: '#5a5c69',
                        borderWidth: 1,
						opacity: 0.85
                    },
                    dimensions: ['City', 'Longitude', 'Latitude', 'Size', 'Color', 'sizeForScatter', 'colorForScatter'],
                    encode: {
                        lng: 1,
                        lat: 2
                    },
                    animationEasing: 'exponentialOut',
                    animationEasingUpdate: 'cubicInOut',
                    animationDurationUpdate: 500,
                    animationDuration: 700,
                    data: []
                }
            ]
        }
    }

    buttonZoomIn(){
        this.zoomIn(1.5);
    }
    buttonZoomOut(){
        this.zoomOut(1.5);
    }

    zoomIn(zoomFactor) {
        let zoom = this.option.geo.zoom;
        zoom = zoom * zoomFactor;
        zoom = Math.max(this.scaleLimit.min, zoom);
        zoom = Math.min(this.scaleLimit.max, zoom);

        this.option.visualMap[0].inRange.symbolSize = [this.dotMinSize*(1+ Math.log(zoom)/2), this.dotMaxSize*(1+ Math.log(zoom)/2)];
        //this.option.visualMap[0].inRange.symbolSize = [this.dotMinSize*growFactor, this.dotMaxSize*growFactor];
        this.option.geo.zoom = zoom;
        this.chart.setOption(this.option);
    }

    zoomOut(zoomFactor) {
        let zoom = this.option.geo.zoom;
        zoom = zoom / zoomFactor;
        zoom = Math.max(this.scaleLimit.min, zoom);
        zoom = Math.min(this.scaleLimit.max, zoom);

        this.option.visualMap[0].inRange.symbolSize = [this.dotMinSize*(1+ Math.log(zoom)/2), this.dotMaxSize*(1+ Math.log(zoom)/2)];
        //this.option.visualMap[0].inRange.symbolSize = [this.dotMinSize*growFactor, this.dotMaxSize*growFactor];

        this.option.geo.zoom = zoom;
        this.chart.setOption(this.option);
    }

    zoomReset() {
        this.option.geo.zoom = 1;
        this.option.visualMap[0].inRange.symbolSize = [this.dotMinSize, this.dotMaxSize];
        this.chart.setOption(this.option);
    }
 
    async loadGeoJson() {
        // Use different map granularity for desktop and mobile users (for performance reasons)
        let germanyStatesJson = await axios.get(`${SITE_URL}/static/assets/geojson/2_germany_states_minimal.json`);
        let germanyCountiesJson = await axios.get(`${SITE_URL}/static/assets/geojson/2_germany_counties_minimal.json`);
        if (!this.isMobile){
            germanyStatesJson = await axios.get(`${SITE_URL}/static/assets/geojson/2_germany_states_tiny.json`);
            germanyCountiesJson = await axios.get(`${SITE_URL}/static/assets/geojson/2_germany_counties_tiny.json`);
        }

        const mixStateDistrictData = germanyCountiesJson.data;

        germanyCountiesJson.data.features.forEach(feature => {
            feature.properties.name = feature.properties['NAME_3'];
        });

        germanyStatesJson.data.features.forEach(feature => {
            feature.properties.name = feature.properties['id'];
            this.geoRegionData.push({
                name: feature.properties['id'],
                itemStyle: {
                    borderColor: '#5a5c69',
                    areaColor: '#fff0',
                    borderWidth: 1,
                    emphasis: {
                        borderColor: '#5a5c69',
                        areaColor: '#fff0',
                        borderWidth: 1
                    }
                },
            });
            mixStateDistrictData.features.push(feature);
        });

        echarts.registerMap('Germany', mixStateDistrictData);

        this.chart.hideLoading();

        this.option.geo.regions = this.geoRegionData;

        // Event for double click on map
		this.chart.on('dblclick', (params) => {
			if (params.componentType === 'geo'){
				this.zoomIn(1.5);
			}
		});

        // Event for moving or zooming map
        this.chart.on('geoRoam', (e) => {
			let zoom = e.zoom;
			if (!isNaN(zoom)){
                let newZoom = this.option.geo.zoom * zoom;
                
                newZoom = Math.max(this.scaleLimit.min, newZoom);
                newZoom = Math.min(this.scaleLimit.max, newZoom);
                this.option.visualMap[0].inRange.symbolSize = [this.dotMinSize*(1 + Math.log(newZoom)/2), this.dotMaxSize*(1+ Math.log(newZoom)/2)];
                this.option.geo.zoom = newZoom;
                this.option.animation = false;
                this.chart.setOption(this.option);
                this.option.animation = true;
			}
		});

        // Event for POI onclick -> Open modal 
        this.chart.on('click', (params) => {
            this.chart.dispatchAction({
                type: 'hideTip'
            });
            this.currentModalLocation = params.dataIndex
            this.buildOrUpdateSummaryModal(true);
            
        });
        this.chart.setOption(this.option);

        this.initLegends();

        this.adaptPOIsize();

        setTimeout(() => this.chart.resize(), 10);

    }

    initLegends() {
        this.colorLegend = new ColorLegend($('#colorLegendMap')[0]);
        this.sizeLegend = new SizeLegend($('#sizeLegendMap')[0]);

        this.colorLegend.setPreviewElement($('#colorLegendPreview')[0]);

        this.colorLegend.setColors(this.colorLeft, this.colorMiddle, this.colorRight);
        this.colorLegend.setRange(this.colorMin, this.colorMax, this.colorMeasure[`name_${this.language}`]);
        this.sizeLegend.setRange(this.sizeMin, this.sizeMax, this.sizeMeasure[`name_${this.language}`]);
    }

    initIndicators() {
        this.colorIndicator = new IndicatorChart($('#colorAverageIndicator')[0]);
        this.sizeIndicator = new IndicatorChart($('#sizeAverageIndicator')[0]);
        this.colorIndicator.setMeasure(this.colorMin, this.colorMax, this.colorMeasure[`name_${this.language}`]);
        this.sizeIndicator.setMeasure(this.sizeMin, this.sizeMax, this.sizeMeasure[`name_${this.language}`]);
        
        this.colorIndicator.setColors(this.colorLeft, this.colorMiddle, this.colorRight);
        this.sizeIndicator.setColors(this.colorLeft, this.colorMiddle, this.colorRight);
    }

    convertData(rawData) {
        return rawData.map((ele, i) => {
            return [
                ele[ `name_${this.language}` ],
                ele.longitude, 
                ele.latitude, 
                ele[this.sizeMeasure.measure_id], 
                ele[this.colorMeasure.measure_id],
                // Replace non-existing with low numbers: Their size is defined by outOfRange in the visualMap
                ele[this.sizeMeasure.measure_id] === '-' ? -1e8 : ele[this.sizeMeasure.measure_id],
                ele[this.colorMeasure.measure_id] === '-' ? -1e8 : ele[this.colorMeasure.measure_id]
            ];
        });
    }

	setPreviewColors(colorLeft, colorMiddle, colorRight) {	
		if (this.colorLegend) this.colorLegend.setPreviewColors(colorLeft, colorMiddle, colorRight);
	}

    setColors(colorLeft, colorMiddle, colorRight) {
        this.colorLeft = colorLeft;
        this.colorRight = colorRight;
        this.colorMiddle = colorMiddle;

        if (colorMiddle == null){
            this.option.visualMap[1].inRange.color = [this.colorLeft, this.colorRight];
        } else {
            this.option.visualMap[1].inRange.color = [this.colorLeft, this.colorMiddle, this.colorRight];
        }

        this.chart.setOption(this.option);

        if (this.colorLegend) this.colorLegend.setColors(colorLeft, colorMiddle, colorRight);
        
        this.colorIndicator.setColors(colorLeft, colorMiddle, colorRight);
        this.sizeIndicator.setColors(colorLeft, colorMiddle, colorRight);
    }
    
    setColorMeasure(measure) {
        super.setColorMeasure(measure);
        let colorMeasureMeta = MEASURES.filter(ele => ele.measure_id === this.colorMeasure.measure_id)[0];
        this.colorMin = colorMeasureMeta.lower_bound;
        this.colorMax = colorMeasureMeta.upper_bound;
        if (this.colorLegend) this.colorLegend.setRange(this.colorMin, this.colorMax, this.colorMeasure[`name_${this.language}`]);

        if (this.colorIndicator) this.colorIndicator.setMeasure(this.colorMin, this.colorMax, this.colorMeasure[`name_${this.language}`]);
    }
    
    setSizeMeasure(measure) {	
        super.setSizeMeasure(measure);
        let sizeMeasureMeta = MEASURES.filter(ele => ele.measure_id === this.sizeMeasure.measure_id)[0];
        this.sizeMin = sizeMeasureMeta.lower_bound;
        this.sizeMax = sizeMeasureMeta.upper_bound;
        if (this.sizeLegend) this.sizeLegend.setRange(this.sizeMin, this.sizeMax, this.sizeMeasure[`name_${this.language}`]);

        if (this.sizeIndicator) this.sizeIndicator.setMeasure(this.sizeMin, this.sizeMax, this.sizeMeasure[`name_${this.language}`]);
    }

    refresh() {	
        this.option.series[0].data = this.data;

        this.option.series[0].dimensions[3] = this.sizeMeasure[`name_${this.language}`];
        this.option.series[0].dimensions[4] = this.colorMeasure[`name_${this.language}`];
        
        this.option.visualMap[0].min = this.sizeMin;
        this.option.visualMap[0].max = this.sizeMax;
        this.option.visualMap[0].text = [Math.ceil(this.sizeMax).toFixed(0), Math.floor(this.sizeMin).toFixed(0)];
        
        this.option.visualMap[1].min = this.colorMin;
        this.option.visualMap[1].max = this.colorMax;
        this.option.visualMap[1].text = [Math.ceil(this.colorMax).toFixed(0), Math.floor(this.colorMin).toFixed(0)];

        this.chart.setOption(this.option);

        this.buildOrUpdateSummaryModal(false);

        if (this.sizeIndicator) this.sizeIndicator.setData(this.data, 3);
        if (this.colorIndicator) this.colorIndicator.setData(this.data, 4);
    }

    buildOrUpdateSummaryModal(openModalIfHidden){	
        if (openModalIfHidden || $('#poiInfoModal').is(":visible")){
            // dataIndex represents map POIs
            let dataIndex = this.currentModalLocation;
            // if it is not defined, the modal content cannot refer to a location and no modal is shown
            if (dataIndex === undefined){
                $('#poiInfoModal').modal('hide');
                return;
            }

            let data = this.rawData[dataIndex];
            // if data is null there is no data entry for this location at this day. 
            // This case only occurs if the date is switched while the modal is open
            // In this case: Keep the current Location name and just update the date
            if (data == null){
                let $modelDataMeasures = $('#modalData_measures');
                $modelDataMeasures.empty();
                let $modelDataPlots = $('#modalData_plots');
                $modelDataPlots.empty();
                let noDataInfo = getI18n('no_data');
                var $noData = 
                    `<div class="row">
                        <span class="col-6">${noDataInfo}</span>
                    </div>
                    `
                $modelDataMeasures.append($noData);
                $modelDataPlots.append($noData);
                const options = { year: 'numeric', month: 'long', day: 'numeric' };
                let dateString = this.language === DE ? this.date.toLocaleDateString('de-DE', options) : this.date.toLocaleDateString('en-EN', options);
                $('#modalDate').text(dateString);
                
            // regular case where data for modal is available
            } else {
                let location_name = data[`name_${this.language}`];
                let desc = data[`description_${this.language}`];

                // Init empty modal
                let $modelContent = $('#poiInfoModalContent');
                $modelContent.empty();

                let date = new Date(data.date);
                const options = { year: 'numeric', month: 'long', day: 'numeric' };
                let dateString = this.language === DE ? date.toLocaleDateString('de-DE', options) : date.toLocaleDateString('en-EN', options);

                // Build POI summary modal Frame
                let $content = $(
                    `   <div id="poiModalSummary">
                            <div class="col-13 d-flex justify-content-between">
                                <span id="modalLocationName" class="" style="font-weight: bold; font-size: large;">${location_name}</span>   
                                <div id="date-slider-btn-map-modal" class="">
                                    <i class="prevDay fas fa-angle-left fa-fw modal-link" title="Previous Day"></i>                                    
                                    <span id="modalDate" class="pr-0 pl-0" style="font-weight: bold; font-size: large; text-align: end;">${dateString}</span>
                                    <i class="nextDay fas fa-angle-right fa-fw modal-link" title="Next Day"></i>
                                </div>
                            </div>
                            <div class="row">
                                <span class="col-12" style="font-size: small;">${desc}</span>
                            </div>
                            <hr/>
                        </div>
                        <div id="poiModalData">
                            <ul class="nav nav-tabs d-flex" role="tablist">
                                <li class="nav-item">
                                    <span class="nav-link active" id="measures-tab" data-toggle="tab" href="#measures" role="tab" aria-controls="measures" aria-selected="true">${getI18n('measures')}</span>
                                </li>
                                <li class="nav-item">
                                    <span class="nav-link" id="plots-tab" data-toggle="tab" href="#plots" role="tab" aria-controls="plots" aria-selected="false">${getI18n('plots')}</span>
                                </li>
                            </ul>
                            <hr class="m-0">
                            <div class="tab-content">
                            <div class="tab-pane fade show active" id="measures" role="tabpanel" aria-labelledby="measures-tab">
                                <div id="modalData_measures" class="card-body pl-2 pr-2 pb-2"></div>
                            </div>
                            <div class="tab-pane fade" id="plots" role="tabpanel" aria-labelledby="plots-tab">
                                <div id="modalData_plots" class="card-body pl-2 pr-2 pb-2"></div>
                            </div>
                            </div>
                        </div>
                    `
                );
                $modelContent.append($content);
                
                $modelContent.find('.prevDay').on('click', () => this.dateSlider.prevDay());
                $modelContent.find('.nextDay').on('click', () => this.dateSlider.nextDay());
                
                // Build modal data content and delete old contents
                let $modelDataMeasures = $('#modalData_measures');
                $modelDataMeasures.empty();
                let $modelDataPlots = $('#modalData_plots');
                $modelDataPlots.empty();

                // Add all measurements
                MEASURES.forEach((measure, cnt) => {
                    let name = measure[`name_${this.language}`];
                    let description = measure[`description_${this.language}`];
                    let value = data[measure.measure_id];
                    $modelDataMeasures.append(
                        `
                        <div class="row align-items-center">
                            <span class="col-5">${name}:</span>
                            <span class="col-4">${value}</span>
                            <span class="col-1">
                                <a href="#" class="fa fa-info-circle" data-placement="top" data-trigger="hover focus" data-toggle="popover" data-original-title="${name}" data-content="${description}"></a>
                            </span>
                            <span class="col-1" id="poiInfoTooltip${cnt}">
                                <a class="btn btn-xs btn-modal-plot show-tooltip shadow-none" title="${getI18n('plot_hist_toggle')}" data-measure="${measure.measure_id}" data-measureName="${name}" data-plotArea="measurePlotArea_${name.replaceAll(' ', '')}" data-toggle="collapse" href="#multiCollapse_${name.replaceAll(' ', '')}" role="button" aria-expanded="false" aria-controls="multiCollapse_${name.replaceAll(' ', '')}"><span class="fas fa-chart-bar"></span></a>
                            </span>
                        </div>
                        <div class="row pb-1" id="poiInfoPlotCollapse${cnt}">
                            <div class="col-12 collapse multi-collapse collapseOne pb-2" id="multiCollapse_${name.replaceAll(' ', '')}">
                                <div class="card card-body p-1">
                                    <div id="measurePlotArea_${name.replaceAll(' ', '')}" class="col-12 p-0">
                                    </div>
                                </div>
                            </div>
                        </div>
                        `
                    );
                });

                // init plot buttons that show the 90 days history for each metric
                $('.btn-modal-plot').click(function () {
                    $(this).toggleClass('active');
                    if (!$(this).hasClass('plotLoaded')){
                        $(this).addClass('plotLoaded');
                        var plotAreaId = '#' + $(this).attr("data-plotArea");
                        var plotMeasureId = $(this).attr("data-measure");
                        var measureName = $(this).attr("data-measureName");
                        var lookbackDays = 90;
                        var titleText = measureName + " - " + lookbackDays + " " + getI18n('hist_title');
                        // differentiate mobile and desktop
                        if ($('#poiModalContent').width() > 500){
                            this.plot = new Plot($(plotAreaId), 20);
                            this.plot.showPlotFromHist(plotMeasureId, measureName, data['location_id'], data['date'], lookbackDays, titleText);
                        } else {
                            this.plot = new Plot($(plotAreaId), 14);
                            this.plot.showPlotFromHist(plotMeasureId, measureName, data['location_id'], data['date'], lookbackDays, titleText);
                        }
                    }
                })

                // Add the section for plots
                $modelDataPlots.append(
                        `
                        <div class="row">
                            <div class="dropdown col-12 pb-3">
                                <button class="btn btn-secondary btn-sm dropdown-toggle" type="button" id="dropdownMenuButton" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
                                    ${getI18n('select_plot')}
                                </button> 
                                <div class="dropdown-menu">
                                </div>
                            </div>                            
                        </div>
                        <div class="row">  
                            <div id="plotArea" class="col-12">
                            </div>                     
                        </div>
                        `
                );

                // Add the plots to the dropdown menu
                if (data.plotData.length > 0){
                    $content.find('.dropdown-menu').append('<h6 class="dropdown-header">Plots</h6>');
                    data.plotData.forEach(plotMeta => {
                        let name = plotMeta[`name_${this.language}`];
                        let plot_id = plotMeta[`plot_id`];
                        $content.find('.dropdown-menu').append(`<a class="dropdown-item" data-value="${plot_id}" href="#">${name}</a>`);
                    });
                } else {
                    $content.find('.dropdown-menu').append('<h6 class="dropdown-header">---</h6>');
                }
                this.initPlots($content.find('#plotArea'));                

                // Show the modal
                $('#poiInfoModal').modal('show');

                // Init the hover legend inside the modal (The small "i" buttons that give the metric descriptions)
                $('[data-toggle="popover"]').popover();
            }
            
        }        
    }

    initPlots($modelContent) {
        $("body").on('click', '.dropdown-menu a', function () {            
            $('#dropdownMenuButton').text($(this).text());
            var plot_id = $(this).data('value');
            if ($('#poiModalContent').width() > 500){
                this.plot = new Plot($modelContent, 25);
                this.plot.showPlotFromId(plot_id);
            } else {
                this.plot = new Plot($modelContent, 15);
                this.plot.showPlotFromId(plot_id);
            }
        });
    }

    convertRemToPixels(rem) {  
        var pixelsize = rem * parseFloat(window.getComputedStyle(document.body).getPropertyValue('font-size'));
        return pixelsize;
    }

    adaptPOIsize(){	
        var width = (window.innerWidth > 0) ? window.innerWidth : screen.width;
        // vars:
        var scaleReferenceMax = 35
        var scaleReferenceMin = 16
        var mobileLayoutBeginsAt = 400
        var intermediateLayoutRem = 25

        // Scale POI for different mobile screen sizes according to scaleReference
        if (width <= mobileLayoutBeginsAt){            
            var newSizeMax = $('#mapChart').width() / (mobileLayoutBeginsAt / scaleReferenceMax);
            var newSizeMin = $('#mapChart').width() / (mobileLayoutBeginsAt / scaleReferenceMin);
            this.dotMaxSize = newSizeMax;
            this.dotMinSize = newSizeMin;
            this.option.visualMap[0].inRange.symbolSize = [this.dotMinSize, this.dotMaxSize];
            this.chart.setOption(this.option);
        // scale POI once legend jumps from site to below map
        } else if ($('#mapChart').height() <= this.convertRemToPixels(intermediateLayoutRem)) {
            if (this.dotMaxSize != 35){
                this.dotMaxSize = 35;
                this.dotMinSize = 15;
                this.option.visualMap[0].inRange.symbolSize = [this.dotMinSize, this.dotMaxSize];
                this.chart.setOption(this.option);
            }
        // original POI size
        } else {
            if (this.dotMaxSize != 45){
                this.dotMaxSize = 45;
                this.dotMinSize = 15;
                this.option.visualMap[0].inRange.symbolSize = [this.dotMinSize, this.dotMaxSize];
                this.chart.setOption(this.option);
            }       
        }       

    }

    resize() {
        let w = $('#mapChart').width();
        if (w === this.chartWidth) return;

        this.adaptPOIsize();
        
        this.chartWidth = w;
        this.chart.resize();

        this.chart.setOption(this.option);

        this.colorLegend.resize();
        this.sizeLegend.resize();

        this.colorIndicator.resize();
        this.sizeIndicator.resize();
    }

    setDateSlider(dateSlider){
        this.dateSlider = dateSlider;
    }

}
