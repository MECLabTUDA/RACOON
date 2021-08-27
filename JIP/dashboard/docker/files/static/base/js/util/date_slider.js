class DateRangeSlider {

    constructor(manager, sliderContext, id, otherIds, language) {
        this.language = language;

        this.id = id;
        this.manager  = manager;
        this.sliderContext = sliderContext;

        this.otherSliders = otherIds.map(function(otherId){ 
            return $(`#date-slider-${otherId}`);
        });
        //this.otherSlider = $(`#date-slider-${otherId}`);

        this.slider = $(`#date-slider-${id}`);
        this.dateLabel = $(`.date-slider-value`);
        this.updateDateRange(this.sliderContext.selectedDate);
        this.initUI();
    }

    updateDateRange(date) {
        this.sliderContext.startDate = new Date(date.getTime());
        this.sliderContext.startDate.setDate(1);

        this.sliderContext.endDate = new Date(this.sliderContext.startDate.getTime());
        this.sliderContext.endDate.setMonth(this.sliderContext.endDate.getMonth() + 1);
        this.sliderContext.endDate.setDate(this.sliderContext.endDate.getDate() - 1);
    }

    getVisibleSteps(d) {
        let last = new Date(d.getFullYear(), d.getMonth() + 1, 0).getDate();

        let w = $(`#date-slider-container-${this.id}`).width();
        if (w <= 500) {
            return [1, 5, 10, 15, 20, 25, last];
        }
        else if (w <= 850) {
            let arr = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, last];
            if (30 < last) arr.push(30);
            return arr;
        }
        else {
            return Array.from({length: last}, (_, index) => index + 1);
        }
    }

    initUI() {
        var dateSlider = this.slider[0];

        noUiSlider.create(dateSlider, {
            start: this.sliderContext.selectedDate.getTime(),
            step:  24 * 60 * 60 * 1000,
            range: {
                'min': this.sliderContext.startDate.getTime(),
                'max': this.sliderContext.endDate.getTime()
            },
        });

        this.refresh();

        // Listening to update event for preview
        dateSlider.noUiSlider.on('update', (values, handle) => {
            var currentValue = new Number(values[handle]);
            let d = new Date(currentValue);
            let dateString = this.language === EN ? dayjs(d).format('MMM DD, YYYY') : dayjs(d).format('DD. MMM YYYY');
            this.dateLabel.text(dateString);
        });

        // Listening to end event for finally apply on chart
        dateSlider.noUiSlider.on('end', (values, handle) => {
            var currentValue = new Number(values[handle]);
            let d = new Date(currentValue);
            this.setDate(d);
        });

        let $buttons = $(`#date-slider-btn-${this.id}`);

        $buttons.find('.prevDay').on('click', () => this.prevDay());
        $buttons.find('.nextDay').on('click', () => this.nextDay());
        $buttons.find('.prevMonth').on('click', () => this.prevMonth());
        $buttons.find('.nextMonth').on('click', () => this.nextMonth());

    }

    addPipListeners() {
        var dateSlider = this.slider[0];

        var pips = dateSlider.querySelectorAll('.noUi-value');
        for (var i = 0; i < pips.length; i++) {
            // For this example. Do this in CSS!
            pips[i].style.cursor = 'pointer';
            pips[i].addEventListener('click', (e) => {
                let val = $(e.target).text();
                let dm = (val === "Today" || val === "Heute") ? new Date().getDate() : Number(val);
                let d = new Date(this.sliderContext.selectedDate.getTime());
                d.setDate(dm);

                dateSlider.noUiSlider.set(d.getTime());
                this.setDate(d);
            });
        }
    }

    refresh() {
        var dateSlider = this.slider[0];

        dateSlider.noUiSlider.updateOptions(
            {
                start: this.sliderContext.selectedDate.getTime(),
                range: {
                    'min': this.sliderContext.startDate.getTime(),
                    'max': this.sliderContext.endDate.getTime()
                },
                pips: {
                    mode: 'steps',
                    density: 3,
                    tooltips: [ true ],
                    format: {
                        to: (value) => {
                            let d = new Date(value);
    
                            let isToday = d.getTime() === today().getTime();
                            let isMob = window.innerWidth <= 800;
    
			                let todayText = getI18n('today');
                            let visibleDays = this.getVisibleSteps(d);
                            let deskString = visibleDays.indexOf(d.getDate()) >= 0 ? ( isToday ? todayText : d.getDate() ) : '';
    
                            return deskString;
                        },
                        from: (value) => {
                            let d = new Date(this.sliderContext.selectedDate.getTime());
                            d.setDate(value);
                            return d;
                        }
                    }
                }
            }, // New options
            false // Boolean 'fireSetEvent'
        );

        this.addPipListeners();
    }

    resize() {
        this.refresh();
    }

    setDate(d) {
        if (d.getTime() === this.sliderContext.selectedDate.getTime()) {
            return;
        }

        if (d.getMonth() !== this.sliderContext.selectedDate.getMonth()) {
            this.updateDateRange(d);
            this.manager.refreshDateSliders();
        }

        let dateString = this.language === EN ? dayjs(d).format('MMM DD, YYYY') : dayjs(d).format('DD. MMM YYYY');
        this.dateLabel.text(dateString);
        this.sliderContext.selectedDate = d;

        this.otherSliders.forEach(function(otherSlider){ otherSlider[0].noUiSlider.set(d.getTime()) });
        //this.otherSlider[0].noUiSlider.set(d.getTime());

        this.manager.setDate(this.sliderContext.selectedDate.getTime());
    }

    prevDay() {
        let d = new Date(this.sliderContext.selectedDate.getTime());
        d.setDate(d.getDate() - 1);

        this.setDate(d);
        this.slider[0].noUiSlider.set(d.getTime());
    }

    nextDay() {
        let d = new Date(this.sliderContext.selectedDate.getTime());
        d.setDate(d.getDate() + 1);

        this.setDate(d);
        this.slider[0].noUiSlider.set(d.getTime());
    }

    prevMonth() {
        let d = new Date(this.sliderContext.selectedDate.getTime());
        d.setDate(1);
        d.setMonth(d.getMonth() - 1);

        this.setDate(d);
        this.slider[0].noUiSlider.set(d.getTime());
    }

    nextMonth() {
        let d = new Date(this.sliderContext.selectedDate.getTime());
        d.setDate(1);
        d.setMonth(d.getMonth() + 1);

        this.setDate(d);
        this.slider[0].noUiSlider.set(d.getTime());
    }
}
