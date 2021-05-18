const inputElementLeft = document.querySelector('.picker-left');
const inputElementRight = document.querySelector('.picker-right');
const inputElementMiddle = document.querySelector('.picker-middle');

//=============================================
// Color picker for left color palette button
//=============================================
const pickerLeft = Pickr.create({
    el: inputElementLeft,
    theme: 'monolith', // or 'monolith', or 'nano'
    default: COLOR_LEFT,
    useAsButton: true,

    swatches: [
        'rgba(244, 67, 54, 1)',
        'rgba(233, 30, 99, 1)',
        'rgba(156, 39, 176, 1)',
        'rgba(103, 58, 183, 1)',
        'rgba(63, 81, 181, 1)',
        'rgba(33, 150, 243, 1)',
        'rgba(3, 169, 244, 1)',
        'rgba(0, 188, 212, 1)',
        'rgba(0, 150, 136, 1)',
        'rgba(76, 175, 80, 1)',
        'rgba(139, 195, 74, 1)',
        'rgba(205, 220, 57, 1)',
        'rgba(255, 235, 59, 1)',
        'rgba(255, 193, 7, 1)',
        COLOR_RIGHT,
		COLOR_LEFT
    ],

    components: {

        // Main components
        preview: true,
        opacity: false,
        hue: true,

        // Input / output Options
        interaction: {
            hex: false,
            rgba: false,
            hsla: false,
            hsva: false,
            cmyk: false,
            input: false,
            cancel: true,
            save: true
        }
    },

	i18n: this.LANGUAGE === "en" ? i18n_en : i18n_de
})
.on('init', instance => {
    $(inputElementLeft).css('color', COLOR_LEFT);
    $(inputElementLeft).css("border-color", COLOR_LEFT); 
})
.on('show', instance => {
	this.startColor = instance.toHEXA().toString().substr(0, 7);
})
.on('save', (color, instance) => {
    instance.hide();
    let cL = color ? color.toHEXA().toString().substr(0, 7) : COLOR_LEFT;
    let cR = pickerRight.getColor() ? pickerRight.getColor().toHEXA().toString().substr(0, 7) : COLOR_RIGHT;
    let cM = pickerMiddle.getColor() ? pickerMiddle.getColor().toHEXA().toString().substr(0, 7) : COLOR_MIDDLE;

    $(inputElementLeft).css('color', cL);
    $(inputElementLeft).css("border-color", cL); 
    chartManager.setColors(cL, useThirdColor(cM), cR);

	//instance.addSwatch(color.toHEXA().toString());
})
.on('hide', instance => {
	let color = instance.getColor();
    let cL = color ? color.toHEXA().toString().substr(0, 7) : COLOR_LEFT;
    let cR = pickerRight.getColor() ? pickerRight.getColor().toHEXA().toString().substr(0, 7) : COLOR_RIGHT;
    let cM = pickerMiddle.getColor() ? pickerMiddle.getColor().toHEXA().toString().substr(0, 7) : COLOR_MIDDLE;

    $(inputElementLeft).css('color', cL);
    $(inputElementLeft).css("border-color", cL); 
    chartManager.setColors(cL, useThirdColor(cM), cR);
})
.on('change', (color, instance) => {
    let cL = color ? color.toHEXA().toString().substr(0, 7) : COLOR_LEFT;
    let cR = pickerRight.getColor() ? pickerRight.getColor().toHEXA().toString().substr(0, 7) : COLOR_RIGHT;
    let cM = pickerMiddle.getColor() ? pickerMiddle.getColor().toHEXA().toString().substr(0, 7) : COLOR_MIDDLE;

    $(inputElementLeft).css('color', cL);
    $(inputElementLeft).css("border-color", cL); 
    chartManager.setPreviewColors(cL, useThirdColor(cM), cR);
})
.on('cancel', instance => {
    instance.hide();
    let cR = pickerRight.getColor() ? pickerRight.getColor().toHEXA().toString().substr(0, 7) : COLOR_RIGHT;
    let cM = pickerMiddle.getColor() ? pickerMiddle.getColor().toHEXA().toString().substr(0, 7) : COLOR_MIDDLE;
	$(inputElementLeft).css('color', this.startColor);
    $(inputElementLeft).css("border-color", this.startColor); 
    chartManager.setColors(this.startColor, useThirdColor(cM), cR);
});


//=============================================
// Color picker for right color palette button
//=============================================
const pickerRight = Pickr.create({
    el: inputElementRight,
    theme: 'monolith', // or 'monolith', or 'nano'
    default: COLOR_RIGHT,
    useAsButton: true,
	adjustableNumbers: true,

    swatches: [
        'rgba(244, 67, 54, 1)',
        'rgba(233, 30, 99, 1)',
        'rgba(156, 39, 176, 1)',
        'rgba(103, 58, 183, 1)',
        'rgba(63, 81, 181, 1)',
        'rgba(33, 150, 243, 1)',
        'rgba(3, 169, 244, 1)',
        'rgba(0, 188, 212, 1)',
        'rgba(0, 150, 136, 1)',
        'rgba(76, 175, 80, 1)',
        'rgba(139, 195, 74, 1)',
        'rgba(205, 220, 57, 1)',
        'rgba(255, 235, 59, 1)',
        'rgba(255, 193, 7, 1)',
        COLOR_RIGHT,
		COLOR_LEFT
    ],

    components: {

        // Main components
        preview: true,
        opacity: false,
        hue: true,

        // Input / output Options
        interaction: {
            hex: false,
            rgba: false,
            hsla: false,
            hsva: false,
            cmyk: false,
            input: false,
            cancel: true,
            save: true
        }
    },

	i18n: this.LANGUAGE === "en" ? i18n_en : i18n_de
})
.on('init', instance => {
    $(inputElementRight).css('color', COLOR_RIGHT);
    $(inputElementRight).css("border-color", COLOR_RIGHT);
})
.on('show', instance => {
	this.startColor = instance.toHEXA().toString().substr(0, 7);
})
.on('change', (color, instance) => {
    let cL = pickerLeft.getColor() ? pickerLeft.getColor().toHEXA().toString().substr(0, 7) : COLOR_LEFT;
    let cR = color ? color.toHEXA().toString().substr(0, 7) : COLOR_RIGHT;
    let cM = pickerMiddle.getColor() ? pickerMiddle.getColor().toHEXA().toString().substr(0, 7) : COLOR_MIDDLE;

    $(inputElementRight).css('color', cR);
    $(inputElementRight).css("border-color", cR);
    chartManager.setPreviewColors(cL, useThirdColor(cM), cR);
})
.on('save', (color, instance) => {
    instance.hide();
    let cL = pickerLeft.getColor() ? pickerLeft.getColor().toHEXA().toString().substr(0, 7) : COLOR_LEFT;
    let cR = color ? color.toHEXA().toString().substr(0, 7) : COLOR_RIGHT;
    let cM = pickerMiddle.getColor() ? pickerMiddle.getColor().toHEXA().toString().substr(0, 7) : COLOR_MIDDLE;

    $(inputElementRight).css('color', cR);
    $(inputElementRight).css("border-color", cR);
    chartManager.setColors(cL, useThirdColor(cM), cR);

	//instance.addSwatch(color.toHEXA().toString());
})
.on('hide', instance => {
	let color = instance.getColor();
    let cL = pickerLeft.getColor() ? pickerLeft.getColor().toHEXA().toString().substr(0, 7) : COLOR_LEFT;
    let cR = color ? color.toHEXA().toString().substr(0, 7) : COLOR_RIGHT;
    let cM = pickerMiddle.getColor() ? pickerMiddle.getColor().toHEXA().toString().substr(0, 7) : COLOR_MIDDLE;

    $(inputElementRight).css('color', cR);
    $(inputElementRight).css("border-color", cR);
    chartManager.setColors(cL, useThirdColor(cM), cR);
})
.on('cancel', instance => {
    instance.hide();
	let cL = pickerLeft.getColor() ? pickerLeft.getColor().toHEXA().toString().substr(0, 7) : COLOR_LEFT;
    let cM = pickerMiddle.getColor() ? pickerMiddle.getColor().toHEXA().toString().substr(0, 7) : COLOR_MIDDLE;
    $(inputElementRight).css('color', this.startColor);
    $(inputElementRight).css("border-color", this.startColor);
    chartManager.setColors(cL, useThirdColor(cM), this.startColor);
});


//=============================================
// Color picker for middle color palette button
//=============================================
const pickerMiddle = Pickr.create({
    el: inputElementMiddle,
    theme: 'monolith', // or 'monolith', or 'nano'
    default: COLOR_MIDDLE,
    useAsButton: true,

    swatches: [
        'rgba(244, 67, 54, 1)',
        'rgba(233, 30, 99, 1)',
        'rgba(156, 39, 176, 1)',
        'rgba(103, 58, 183, 1)',
        'rgba(63, 81, 181, 1)',
        'rgba(33, 150, 243, 1)',
        'rgba(3, 169, 244, 1)',
        'rgba(0, 188, 212, 1)',
        'rgba(0, 150, 136, 1)',
        'rgba(76, 175, 80, 1)',
        'rgba(139, 195, 74, 1)',
        'rgba(205, 220, 57, 1)',
        'rgba(255, 235, 59, 1)',
        'rgba(255, 193, 7, 1)',
        COLOR_RIGHT,
		COLOR_LEFT,
        COLOR_MIDDLE
    ],

    components: {

        // Main components
        preview: true,
        opacity: false,
        hue: true,

        // Input / output Options
        interaction: {
            hex: false,
            rgba: false,
            hsla: false,
            hsva: false,
            cmyk: false,
            input: false,
            cancel: true,
            save: true
        }
    },

	i18n: this.LANGUAGE === "en" ? i18n_en : i18n_de
})
.on('init', instance => {
    $(inputElementMiddle).css('color', COLOR_MIDDLE);
    $(inputElementMiddle).css("border-color", COLOR_MIDDLE); 
})
.on('show', instance => {
	this.startColor = instance.toHEXA().toString().substr(0, 7);
})
.on('save', (color, instance) => {
    instance.hide();
    let cL = pickerLeft.getColor() ? pickerLeft.getColor().toHEXA().toString().substr(0, 7) : COLOR_LEFT;
    let cR = pickerRight.getColor() ? pickerRight.getColor().toHEXA().toString().substr(0, 7) : COLOR_RIGHT;
    let cM = color ? color.toHEXA().toString().substr(0, 7) : COLOR_MIDDLE;

    $(inputElementMiddle).css('color', cM);
    $(inputElementMiddle).css("border-color", cM); 
    chartManager.setColors(cL, useThirdColor(cM), cR);

	//instance.addSwatch(color.toHEXA().toString());
})
.on('hide', instance => {
	let color = instance.getColor();
    let cL = pickerLeft.getColor() ? pickerLeft.getColor().toHEXA().toString().substr(0, 7) : COLOR_LEFT;
    let cR = pickerRight.getColor() ? pickerRight.getColor().toHEXA().toString().substr(0, 7) : COLOR_RIGHT;
    let cM = color ? color.toHEXA().toString().substr(0, 7) : COLOR_MIDDLE;

    $(inputElementMiddle).css('color', cM);
    $(inputElementMiddle).css("border-color", cM); 
    chartManager.setColors(cL, useThirdColor(cM), cR);
})
.on('change', (color, instance) => {
    let cL = pickerLeft.getColor() ? pickerLeft.getColor().toHEXA().toString().substr(0, 7) : COLOR_LEFT;
    let cR = pickerRight.getColor() ? pickerRight.getColor().toHEXA().toString().substr(0, 7) : COLOR_RIGHT;
    let cM = color ? color.toHEXA().toString().substr(0, 7) : COLOR_MIDDLE;

    $(inputElementMiddle).css('color', cM);
    $(inputElementMiddle).css("border-color", cM); 
    chartManager.setPreviewColors(cL, useThirdColor(cM), cR);
})
.on('cancel', instance => {
    instance.hide();
    let cL = pickerLeft.getColor() ? pickerLeft.getColor().toHEXA().toString().substr(0, 7) : COLOR_LEFT;
    let cR = pickerRight.getColor() ? pickerRight.getColor().toHEXA().toString().substr(0, 7) : COLOR_RIGHT;
	$(inputElementMiddle).css('color', this.startColor);
    $(inputElementMiddle).css("border-color", this.startColor); 
    chartManager.setColors(cL, useThirdColor(this.startColor), cR);
});


function useThirdColor(input_color){
    var is_used = $("#useThirdColorCheckbox").prop('checked');
    if (is_used){
        return input_color;
    } else {
        return null;
    }
}

// onclick function for "use third color" checkbox
$('#useThirdColorCheckbox').change(function() {
    let cL = pickerLeft.getColor() ? pickerLeft.getColor().toHEXA().toString().substr(0, 7) : COLOR_LEFT;
    let cR = pickerRight.getColor() ? pickerRight.getColor().toHEXA().toString().substr(0, 7) : COLOR_RIGHT;
    let cM = pickerMiddle.getColor() ? pickerMiddle.getColor().toHEXA().toString().substr(0, 7) : COLOR_MIDDLE;
    var isChecked = $(this).prop('checked');
    if (isChecked) {
        chartManager.setColors(cL, cM, cR);
        $('.picker-middle').show();
    } else {
        chartManager.setColors(cL, null, cR);
        $('.picker-middle').hide();
    }
})

function loadPreset(colorLeft, colorMiddle, colorRight){
    $(inputElementRight).css('color', colorRight);
    $(inputElementRight).css("border-color", colorRight);
    pickerRight.setColor(colorRight, silent=true);
    $(inputElementLeft).css('color', colorLeft);
    $(inputElementLeft).css("border-color", colorLeft);
    pickerLeft.setColor(colorLeft, silent=true);
    if (colorMiddle == null){
        $('#useThirdColorCheckbox').prop('checked', false);
        $('.picker-middle').hide();
    } else {
        $(inputElementMiddle).css('color', colorMiddle);
        $(inputElementMiddle).css("border-color", colorMiddle);
        pickerMiddle.setColor(colorMiddle, silent=true);
        $('#useThirdColorCheckbox').prop('checked', true);
        $('.picker-middle').show();
    }
    chartManager.setColors(colorLeft, colorMiddle, colorRight);
}

// Definition of color presets for the color selection menu
var gradientPresets = [
    {"cL": "#ed2415", "cM": "#ffd607", "cR": "#43d149"},
    {"cL": "#f2f21b", "cM": null, "cR": "#f44336"},
    {"cL": "#FFFF33", "cM": "#66FF66", "cR": "#3399FF"},
    {"cL": "#F5F5F5", "cM": null, "cR": "#EE7218"},
];

// Create the color presets from the previous definitions
$('.gradientPreset').each(function(i) {
    cL = gradientPresets[i]['cL'];
    cM = gradientPresets[i]['cM'];
    cR = gradientPresets[i]['cR'];
    if (cM == null){
        $(this).css('background-image', `linear-gradient(to right, ${gradientPresets[i]['cL']}, ${gradientPresets[i]['cR']})`);
    } else {
        $(this).css('background-image', `linear-gradient(to right, ${gradientPresets[i]['cL']}, ${gradientPresets[i]['cM']},${gradientPresets[i]['cR']})`);
    }
    $(this).on('click', e => {
        loadPreset(gradientPresets[i]['cL'], gradientPresets[i]['cM'], gradientPresets[i]['cR'])
    });
});