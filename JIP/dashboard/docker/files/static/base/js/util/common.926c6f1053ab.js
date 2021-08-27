function today() {
    let d = new Date();
    d.setHours(0);
    d.setMinutes(0);
    d.setSeconds(0);
    d.setMilliseconds(0);

    return d;
}

const SITE_URL = `${window.location.protocol}//${window.location.host}`;
const API_URL = `${SITE_URL}/api`;

const MAP_CHART = 0;
const TABLE_CHART = 1;
const AGGREGATION_CHART = 2;
const LOCATIONS = {};
const MEASURES = [];
const EN = 'en';
const DE = 'de';

const DECIMALS = 4;
const DECIMALS_AGG = 2;

const COLOR_LEFT = "#F5F5F5";
const COLOR_RIGHT = "#EE7218";
const COLOR_MIDDLE = "#F2B487";

class CovidChart {

    constructor(colorMeasure, sizeMeasure, language) {
        this.language = language;
        this.date = null;
        this.rawData = [];
        this.data = [];
        
        this.colorLeft = COLOR_LEFT;
        this.colorRight = COLOR_RIGHT;
        this.setSizeMeasure(sizeMeasure);
        this.setColorMeasure(colorMeasure);
    }

    setColors(colorLeft, colorMiddle, colorRight, ) {
        this.colorLeft = colorLeft;
        this.colorRight = colorRight;
        this.colorMiddle = colorMiddle;
    }

    setColorMeasure(measure) {
        this.colorMeasure = measure;
    }

    setSizeMeasure(measure) {
        this.sizeMeasure = measure;
    }

    setData(date, data, plotMetaData) {
        this.date = date;
        this.rawData = data;
        this.data = this.convertData(data);
        this.plotMetaData = plotMetaData;
    }

    convertData(rawData) {
        // Override to transform data
        return [ ...rawData ];
    }

    refresh() {
        // Override
    }

    resize() {
        // Override
    }

    setDateSlider(dateSlider){
        // Override
    }

}

$(document).ready(function() {
    $('[data-toggle="tooltip"]').tooltip();
    $('#loginModalButton').tooltip();
    $('#profileModalButton').tooltip();
});

function getCookie(name) {
    var cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        var cookies = document.cookie.split(';');
        for (var i = 0; i < cookies.length; i++) {
            var cookie = jQuery.trim(cookies[i]);
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}
var csrftoken = getCookie('csrftoken');


// Logic for login:
$('.login_form').on('submit', function(e) {
    $('.label_login_fail').hide();
    $('.label_login_success').hide();

    e.preventDefault();
    var data = $('.login_form').serialize();
    $.ajax({
        type: 'POST',
        url: `${ SITE_URL }/user_login/`,
        headers:{
            "X-CSRFToken": getCookie('csrftoken')
        },
        data: data,
        success: function (data) {
            if (data['success']){
                $('#tokenArea').text(data['token']);
                $('.label_login_fail').hide();
                $('.label_login_success').show();
                $('#loginModal').modal('hide');
                $('#profile-username').text($(".login_form input[name=username]").val());
                chartManager.reloadAll();

                $('#notification-box').text("Logged In");
                $('#notification-box').addClass('show').delay(1500).queue(function() {
                    $('#profileModalButton').show();
                    $('#loginModalButton').hide();
                    $('.label_login_success').hide();
                    $(this).removeClass('show').dequeue();
                });
                
            } else {
                $('.label_login_fail').show();
            }
        }, 
        error: function (data) {
            $('.label_login_fail').show();
        }
    })
})

// Logic for logout:
$('.profile_form').on('submit', function(e) {
    e.preventDefault(); // prevent form from reloading page

    $.ajax({
        type: 'POST',
        url: `${ SITE_URL }/user_logout/`,
        headers:{
            "X-CSRFToken": getCookie('csrftoken')
        },
        success: function (data) {
            if (data['success']){
                $('#tokenArea').text('---');
                $('#profileModal').modal('hide');
                $('#profile-username').text("");
                chartManager.reloadAll();

                $('#notification-box').text("Logged Out");
                $('#notification-box').addClass('show').delay(1500).queue(function() {
                    $('#profileModalButton').hide();
                    $('#loginModalButton').show();
                    $('input[name=password').val('');
                    $(this).removeClass('show').dequeue();
                });
            }
        }, 
    })
});

$('#filterModalButton').on('click', function(e) {
    chartManager.reloadAll();
});