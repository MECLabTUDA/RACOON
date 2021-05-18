var tour = introJs();
var study = introJs();

var stepsTour = [
    {// 1
        intro: getI18n('tour:start'),
    },
    {// 2
        element:document.querySelector('#mapChart'),
        title: getI18n('tour:mapView_title'),
        intro: getI18n('tour:mapView'),
        position: 'auto'
    },
    {// 3
        element:document.querySelector('#poiDemoModalContent'),
        title: getI18n('tour:poi_title'),
        intro: getI18n('tour:poi'),
        position: 'auto'
    },
    {// 4
        element:document.querySelector('#poiDemoModalSummary'),
        title: getI18n('tour:poiDetails_title'),
        intro: getI18n('tour:poiDetails'),
        position: 'auto'
    },
    {// 5
        element:document.querySelector('#poiDemoModalData'),
        title: getI18n('tour:poiData_title'),
        intro: getI18n('tour:poiData'),
        position: 'auto'
        },
    {// 6
        element:document.querySelector('#poiDemoTooltipBtn'),
        title: getI18n('tour:poiInfo_title'),
        intro: getI18n('tour:poiInfo'),
        position: 'left'
    },
    {// 7
        element:document.querySelector('#poiDemoHistoryBtn'),
        title: getI18n('tour:poiHistory_title'),
        intro: getI18n('tour:poiHistory'),
        position: 'auto'
    },
    {// 8
        element:document.querySelector('#data-tab'),
        title: getI18n('tour:table_title'),
        intro: getI18n('tour:table'),
    },
    {// 9
        element:document.querySelector('#aggregation-tab'),
        title: getI18n('tour:aggregation_title'),
        intro: getI18n('tour:aggregation'),
    },
    {// 10
        element:document.querySelector('#dateSliderCard'),
        title: getI18n('tour:dateSlider_title'),
        intro: getI18n('tour:dateSlider'),
    },
    {// 11
        element:document.querySelector('#legendCard'),
        title: getI18n('tour:legend_title'),
        intro: getI18n('tour:legend'),
    },
    {// 12
        element:document.querySelector('#sizeLegendMap'),
        title: getI18n('tour:legendSize_title'),
        intro: getI18n('tour:legendSize'),
    },
    {// 13
        element:document.querySelector('#colorLegendMap'),
        title: getI18n('tour:legendColor_title'),
        intro: getI18n('tour:legendColor'),
    },
    {// 14
        element:document.querySelector('#indicatorCard'),
        title: getI18n('tour:indicators_title'),
        intro: getI18n('tour:indicators'),
    },
    {// 15
        element:document.querySelector('#btnSettingsModal'),
        title: getI18n('tour:settings_title'),
        intro: getI18n('tour:settings'),
    },
    {// 16
        element:document.querySelector('#settingsModalContent'),
        title: getI18n('tour:settingsDetails_title'),
        intro: getI18n('tour:settingsDetails'),
    },
    {// 17
        element:document.querySelector('#btnFilterModal'),
        title: getI18n('tour:filter_title'),
        intro: getI18n('tour:filter'),
    },
    {// 18
        element:document.querySelector('#filterModalContent'),
        title: getI18n('tour:filterDetails_title'),
        intro: getI18n('tour:filterDetails'),
    },
    {// 19
        element:document.querySelector('#btnColorModal'),
        title: getI18n('tour:color_title'),
        intro: getI18n('tour:color'),
    },
    {// 20
        element:document.querySelector('#scolorModalContent'),
        title: getI18n('tour:colorDetails_title'),
        intro: getI18n('tour:colorDetails'),
    },
    {// 21
        element:document.querySelector('#colorSelectorLeft'),
        title: getI18n('tour:colorBtn_title'),
        intro: getI18n('tour:colorBtn'),
    },
    //{// 22
    //    element: document.querySelector('#navLanguageSwitch'),
    //    title: getI18n('tour:language_title'),
    //    intro: getI18n('tour:language'),
    //    position: 'bottom-middle-aligned'
    //}, 
    {// 23
        element: document.querySelector('#navThemeSwitch'),
        title: getI18n('tour:theme_title'),
        intro: getI18n('tour:theme'),
    },  
    {// 24
        element: document.querySelector('#navLoginLogout'),
        title: getI18n('tour:login_title'),
        intro: getI18n('tour:login'),
    }, 

    {// 25
        title: getI18n('tour:end_title'),
        intro: getI18n('tour:end')
    },
];

var stepsStudy = [{intro: getI18n('tour:study')}].concat(stepsTour);


tour.setOptions({
    exitOnOverlayClick: false,
    showProgress: true,
    showBullets: false,
    disableInteraction: true,
    positionPrecedence: ['top', 'bottom', 'left', 'right'],
    steps: stepsTour,
});

study.setOptions({
    exitOnOverlayClick: false,
    showProgress: true,
    showBullets: false,
    disableInteraction: true,
    positionPrecedence: ['top', 'bottom', 'left', 'right'],
    steps: stepsStudy,
});

tour.onbeforechange(async function () {
    if (this._currentStep === 2){
        $('#poiDemoModal').removeClass('fade');
        $('#poiDemoModal').modal('show');
    }
    if (this._currentStep === 7){
        $('#poiDemoModal').addClass('fade');
        $('#poiDemoModal').modal('hide')
        setTimeout(function(){            
            window.dispatchEvent(new Event('resize'));
        }, 500);      
    }

    if (this._currentStep === 15) {
        $('#settingsModal').removeClass('fade');
        $('#settingsModal').modal('show');
        setTimeout(function(){            
            window.dispatchEvent(new Event('resize'));
        }, 500);       
    }
    if (this._currentStep === 16) {
        $('#settingsModal').addClass('fade');
        $('#settingsModal').modal('hide');
    }

    if (this._currentStep === 17) {
        $('#filterModal').removeClass('fade');
        $('#filterModal').modal('show');
        setTimeout(function(){            
            window.dispatchEvent(new Event('resize'));
        }, 500);      
    }
    if (this._currentStep === 18) {
        $('#filterModal').addClass('fade');
        $('#filterModal').modal('hide');
    }  

    if (this._currentStep === 19) {
        $('#colorModal').removeClass('fade');
        $('#colorModal').modal('show');
        setTimeout(function(){            
            window.dispatchEvent(new Event('resize'));
        }, 500);      
    }
    if (this._currentStep === 21) {
        $('#colorModal').addClass('fade');
        $('#colorModal').modal('hide');
    }  
    return true;  
});

study.onbeforechange(async function () {
    if (this._currentStep === 3){
        $('#poiDemoModal').removeClass('fade');
        $('#poiDemoModal').modal('show');
    }
    if (this._currentStep === 8){
        $('#poiDemoModal').addClass('fade');
        $('#poiDemoModal').modal('hide')
        setTimeout(function(){            
            window.dispatchEvent(new Event('resize'));
        }, 500);      
    }

    if (this._currentStep === 16) {
        $('#settingsModal').removeClass('fade');
        $('#settingsModal').modal('show');
        setTimeout(function(){            
            window.dispatchEvent(new Event('resize'));
        }, 500);       
    }
    if (this._currentStep === 17) {
        $('#settingsModal').addClass('fade');
        $('#settingsModal').modal('hide');
    }

    if (this._currentStep === 18) {
        $('#filterModal').removeClass('fade');
        $('#filterModal').modal('show');
        setTimeout(function(){            
            window.dispatchEvent(new Event('resize'));
        }, 500);      
    }
    if (this._currentStep === 19) {
        $('#filterModal').addClass('fade');
        $('#filterModal').modal('hide');
    }  

    if (this._currentStep === 20) {
        $('#colorModal').removeClass('fade');
        $('#colorModal').modal('show');
        setTimeout(function(){            
            window.dispatchEvent(new Event('resize'));
        }, 500);      
    }
    if (this._currentStep === 22) {
        $('#colorModal').addClass('fade');
        $('#colorModal').modal('hide');
    }  
    return true;  
});

const queryString = window.location.search;
const urlParams = new URLSearchParams(queryString);
if (urlParams.has('study')){
    setTimeout(function(){            
        window.dispatchEvent(new Event('resize'));
    }, 200);  
    study.start();
}else if (urlParams.has('tour')){
    setTimeout(function(){            
        window.dispatchEvent(new Event('resize'));
    }, 200);  
    tour.start();
}

var startbtn = $('#startTourBtn');
startbtn.on('click', function(e){
    e.preventDefault();
    tour.start();
})

