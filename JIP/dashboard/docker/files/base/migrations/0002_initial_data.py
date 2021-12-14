from django.db import migrations, models
from datetime import date, timedelta
import random
import os

class Migration(migrations.Migration):

    dependencies = [
        ('base', '0001_initial'),
    ]

    # five different plots (for example usage)
    plotA = """{"grid": {"left": 50, "top": 35, "right": 55, "bottom": 40}, "xAxis": {"type": "value", "name": "Epoch"}, "yAxis": {"name": "Loss", "type": "value"}, "series": [{"data": [[0, 800], [1, 500], [2, 300], [3, 180], [4, 110], [5, 60], [6, 40], [7, 35], [8, 34], [9, 34], [9, 33], [10, 32]], "type": "line", "smooth": true}]}"""

    plotB = """{"tooltip": {"trigger": "axis", "axisPointer": {"type": "cross", "crossStyle": {"color": "#999"}}}, "toolbox": {"feature": {"dataView": {"show": true, "readOnly": false}, "magicType": {"show": true, "type": ["line", "bar"]}, "saveAsImage": {"show": true}}}, "legend": {"top": 30, "data": ["Item A", "Item B", "Item C"]}, "grid": {"left": 50, "top": 90, "right": 50, "bottom": 20}, "xAxis": [{"type": "category", "data": ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"], "axisPointer": {"type": "shadow"}}], "yAxis": [{"type": "value", "name": "Rain", "min": 0, "max": 250, "interval": 50, "axisLabel": {"formatter": "{value} ml"}}, {"type": "value", "name": "Temperature", "min": 0, "max": 25, "interval": 5, "axisLabel": {"formatter": "{value} °C"}}], "series": [{"name": "Item A", "type": "bar", "data": [2.0, 4.9, 7.0, 23.2, 25.6, 76.7, 135.6, 162.2, 32.6, 20.0, 6.4, 3.3]}, {"name": "Item B", "type": "bar", "data": [2.6, 5.9, 9.0, 26.4, 28.7, 70.7, 175.6, 182.2, 48.7, 18.8, 6.0, 2.3]}, {"name": "Item C", "type": "line", "yAxisIndex": 1, "data": [2.0, 2.2, 3.3, 4.5, 6.3, 10.2, 20.3, 23.4, 23.0, 16.5, 12.0, 6.2]}]}"""

    plotC = """{"legend": {"top": "bottom"}, "toolbox": {"show": true, "feature": {"mark": {"show": true}, "dataView": {"show": true, "readOnly": false}, "saveAsImage": {"show": true}}}, "series": [{"name": "Segmentations", "type": "pie", "radius": ["10%", "70%"], "center": ["50%", "50%"], "roseType": "area", "itemStyle": {"borderRadius": 8}, "bottom": 50, "data": [{"value": 40, "name": "Heart"}, {"value": 38, "name": "Lung"}, {"value": 32, "name": "Liver"}, {"value": 30, "name": "Brain"}, {"value": 28, "name": "Kidney"}, {"value": 26, "name": "Knee"}, {"value": 22, "name": "Eye"}, {"value": 18, "name": "Ear"}]}]}"""

    plotD = """{"tooltip": {"position": "top"}, "visualMap": {"type": "piecewise", "top": "middle", "min": 0, "max": 6, "left": 0, "splitNumber": 6, "dimension": 2, "pieces": [{"value": 0, "label": "cluster 0", "color": "#37A2DA"}, {"value": 1, "label": "cluster 1", "color": "#e06343"}, {"value": 2, "label": "cluster 2", "color": "#37a354"}, {"value": 3, "label": "cluster 3", "color": "#b55dba"}, {"value": 4, "label": "cluster 4", "color": "#b5bd48"}, {"value": 5, "label": "cluster 5", "color": "#8378EA"}, {"value": 6, "label": "cluster 6", "color": "#96BFFF"}]}, "grid": {"top": 20, "bottom": 25, "right": 10, "left": 110}, "xAxis": {}, "yAxis": {}, "series": {"type": "scatter", "encode": {"tooltip": [0, 1]}, "symbolSize": 15, "itemStyle": {"borderColor": "#555"}, "data": [[3.275154, 2.957587, 5], [-3.344465, 2.603513, 4], [0.355083, -3.376585, 1], [1.852435, 3.547351, 5], [-2.078973, 2.552013, 4], [-0.993756, -0.884433, 3], [2.682252, 4.007573, 5], [-3.087776, 2.878713, 4], [-1.565978, -1.256985, 3], [2.441611, 0.444826, 0], [-0.659487, 3.111284, 4], [-0.459601, -2.618005, 3], [2.17768, 2.387793, 0], [-2.920969, 2.917485, 4], [-0.028814, -4.168078, 1], [3.625746, 2.119041, 0], [-3.912363, 1.325108, 4], [-0.551694, -2.814223, 3], [2.855808, 3.483301, 5], [-3.594448, 2.856651, 4], [0.421993, -2.372646, 1], [1.650821, 3.407572, 5], [-2.082902, 3.384412, 4], [-0.718809, -2.492514, 3], [4.513623, 3.841029, 5], [-4.822011, 4.607049, 2], [-0.656297, -1.449872, 3], [1.919901, 4.439368, 5], [-3.287749, 3.918836, 4], [-1.576936, -2.977622, 3], [3.598143, 1.97597, 0], [-3.977329, 4.900932, 2], [-1.79108, -2.184517, 3], [3.914654, 3.559303, 5], [-1.910108, 4.166946, 4], [-1.226597, -3.317889, 3], [1.148946, 3.345138, 5], [-2.113864, 3.548172, 4], [0.845762, -3.589788, 1], [2.629062, 3.535831, 5], [-1.640717, 2.990517, 4], [-1.881012, -2.485405, 3], [4.606999, 3.510312, 5], [-4.366462, 4.023316, 2], [0.765015, -3.00127, 1], [3.121904, 2.173988, 0], [-4.025139, 4.65231, 2], [-0.559558, -3.840539, 1], [4.376754, 4.863579, 5], [-1.874308, 4.032237, 4], [-0.089337, -3.026809, 1], [3.997787, 2.518662, 0], [-3.082978, 2.884822, 4], [0.845235, -3.454465, 1], [1.327224, 3.358778, 5], [-2.889949, 3.596178, 4], [-0.966018, -2.839827, 3], [2.960769, 3.079555, 5], [-3.275518, 1.577068, 4], [0.639276, -3.41284, 1]]}}"""

    plotE = """{"tooltip": {}, "series": [{"name": "Les Miserables", "type": "graph", "layout": "none", "data": [{"id": "0", "name": "Myriel", "symbolSize": 10.12381, "x": -266.82776, "y": 299.6904, "value": 28.685715, "category": 0}, {"id": "1", "name": "Napoleon", "symbolSize": 2.6666666666666665, "x": -418.08344, "y": 446.8853, "value": 4, "category": 0}, {"id": "2", "name": "MlleBaptistine", "symbolSize": 6.323809333333333, "x": -212.76357, "y": 245.29176, "value": 9.485714, "category": 1}, {"id": "3", "name": "MmeMagloire", "symbolSize": 6.323809333333333, "x": -242.82404, "y": 235.26283, "value": 9.485714, "category": 1}, {"id": "4", "name": "CountessDeLo", "symbolSize": 2.6666666666666665, "x": -379.30386, "y": 429.06424, "value": 4, "category": 0}, {"id": "5", "name": "Geborand", "symbolSize": 2.6666666666666665, "x": -417.26337, "y": 406.03506, "value": 4, "category": 0}, {"id": "6", "name": "Champtercier", "symbolSize": 2.6666666666666665, "x": -332.6012, "y": 485.16974, "value": 4, "category": 0}, {"id": "7", "name": "Cravatte", "symbolSize": 2.6666666666666665, "x": -382.69568, "y": 475.09113, "value": 4, "category": 0}, {"id": "8", "name": "Count", "symbolSize": 2.6666666666666665, "x": -320.384, "y": 387.17325, "value": 4, "category": 0}, {"id": "9", "name": "OldMan", "symbolSize": 2.6666666666666665, "x": -344.39832, "y": 451.16772, "value": 4, "category": 0}, {"id": "10", "name": "Labarre", "symbolSize": 2.6666666666666665, "x": -89.34107, "y": 234.56128, "value": 4, "category": 1}, {"id": "11", "name": "Valjean", "symbolSize": 15.66666666666667, "x": -87.93029, "y": -6.8120565, "value": 100, "category": 1}, {"id": "12", "name": "Marguerite", "symbolSize": 4.495239333333333, "x": -339.77908, "y": -184.69139, "value": 6.742859, "category": 1}, {"id": "13", "name": "MmeDeR", "symbolSize": 2.6666666666666665, "x": -194.31313, "y": 178.55301, "value": 4, "category": 1}, {"id": "14", "name": "Isabeau", "symbolSize": 2.6666666666666665, "x": -158.05168, "y": 201.99768, "value": 4, "category": 1}, {"id": "15", "name": "Gervais", "symbolSize": 2.6666666666666665, "x": -127.701546, "y": 242.55057, "value": 4, "category": 1}, {"id": "16", "name": "Tholomyes", "symbolSize": 10.295237333333333, "x": -385.2226, "y": -393.5572, "value": 25.942856, "category": 2}, {"id": "17", "name": "Listolier", "symbolSize": 10.638097333333334, "x": -516.55884, "y": -393.98975, "value": 20.457146, "category": 2}, {"id": "18", "name": "Fameuil", "symbolSize": 10.638097333333334, "x": -464.79382, "y": -493.57944, "value": 20.457146, "category": 2}, {"id": "19", "name": "Blacheville", "symbolSize": 10.638097333333334, "x": -515.1624, "y": -456.9891, "value": 20.457146, "category": 2}, {"id": "20", "name": "Favourite", "symbolSize": 10.638097333333334, "x": -408.12122, "y": -464.5048, "value": 20.457146, "category": 2}, {"id": "21", "name": "Dahlia", "symbolSize": 10.638097333333334, "x": -456.44113, "y": -425.13303, "value": 20.457146, "category": 2}, {"id": "22", "name": "Zephine", "symbolSize": 10.638097333333334, "x": -459.1107, "y": -362.5133, "value": 20.457146, "category": 2}, {"id": "23", "name": "Fantine", "symbolSize": 12.266666666666666, "x": -313.42786, "y": -289.44803, "value": 42.4, "category": 2}, {"id": "24", "name": "MmeThenardier", "symbolSize": 10.95238266666667, "x": 4.6313396, "y": -273.8517, "value": 31.428574, "category": 7}, {"id": "25", "name": "Thenardier", "symbolSize": 12.095235333333335, "x": 82.80825, "y": -203.1144, "value": 45.142853, "category": 7}, {"id": "26", "name": "Cosette", "symbolSize": 10.95238266666667, "x": 78.64646, "y": -31.512747, "value": 31.428574, "category": 6}, {"id": "27", "name": "Javert", "symbolSize": 12.923806666666668, "x": -81.46074, "y": -204.20204, "value": 47.88571, "category": 7}, {"id": "28", "name": "Fauchelevent", "symbolSize": 8.152382000000001, "x": -225.73984, "y": 82.41631, "value": 12.228573, "category": 4}, {"id": "29", "name": "Bamatabois", "symbolSize": 10.466666666666667, "x": -385.6842, "y": -20.206686, "value": 23.2, "category": 3}, {"id": "30", "name": "Perpetue", "symbolSize": 4.495239333333333, "x": -403.92447, "y": -197.69823, "value": 6.742859, "category": 2}, {"id": "31", "name": "Simplice", "symbolSize": 8.152382000000001, "x": -281.4253, "y": -158.45137, "value": 12.228573, "category": 2}, {"id": "32", "name": "Scaufflaire", "symbolSize": 2.6666666666666665, "x": -122.41348, "y": 210.37503, "value": 4, "category": 1}, {"id": "33", "name": "Woman1", "symbolSize": 4.495239333333333, "x": -234.6001, "y": -113.15067, "value": 6.742859, "category": 1}, {"id": "34", "name": "Judge", "symbolSize": 10.809524666666666, "x": -387.84915, "y": 58.7059, "value": 17.714287, "category": 3}, {"id": "35", "name": "Champmathieu", "symbolSize": 9.809524666666666, "x": -338.2307, "y": 87.48405, "value": 17.714287, "category": 3}, {"id": "36", "name": "Brevet", "symbolSize": 9.809524666666666, "x": -453.26874, "y": 58.94648, "value": 17.714287, "category": 3}, {"id": "37", "name": "Chenildieu", "symbolSize": 9.809524666666666, "x": -386.44904, "y": 140.05937, "value": 17.714287, "category": 3}, {"id": "38", "name": "Cochepaille", "symbolSize": 9.809524666666666, "x": -446.7876, "y": 123.38005, "value": 17.714287, "category": 3}, {"id": "39", "name": "Pontmercy", "symbolSize": 6.323809333333333, "x": 336.49738, "y": -269.55914, "value": 9.485714, "category": 6}, {"id": "40", "name": "Boulatruelle", "symbolSize": 2.6666666666666665, "x": 29.187843, "y": -460.13132, "value": 4, "category": 7}, {"id": "41", "name": "Eponine", "symbolSize": 10.95238266666667, "x": 238.36697, "y": -210.00926, "value": 31.428574, "category": 7}, {"id": "42", "name": "Anzelma", "symbolSize": 6.323809333333333, "x": 189.69513, "y": -346.50662, "value": 9.485714, "category": 7}, {"id": "43", "name": "Woman2", "symbolSize": 6.323809333333333, "x": -187.00418, "y": -145.02663, "value": 9.485714, "category": 6}, {"id": "44", "name": "MotherInnocent", "symbolSize": 4.495239333333333, "x": -252.99521, "y": 129.87549, "value": 6.742859, "category": 4}, {"id": "45", "name": "Gribier", "symbolSize": 2.6666666666666665, "x": -296.07935, "y": 163.11964, "value": 4, "category": 4}, {"id": "46", "name": "Jondrette", "symbolSize": 2.6666666666666665, "x": 550.3201, "y": 522.4031, "value": 4, "category": 5}, {"id": "47", "name": "MmeBurgon", "symbolSize": 4.495239333333333, "x": 488.13535, "y": 356.8573, "value": 6.742859, "category": 5}, {"id": "48", "name": "Gavroche", "symbolSize": 12.06667066666667, "x": 387.89572, "y": 110.462326, "value": 61.600006, "category": 8}, {"id": "49", "name": "Gillenormand", "symbolSize": 10.638097333333334, "x": 126.4831, "y": 68.10622, "value": 20.457146, "category": 6}, {"id": "50", "name": "Magnon", "symbolSize": 4.495239333333333, "x": 127.07365, "y": -113.05923, "value": 6.742859, "category": 6}, {"id": "51", "name": "MlleGillenormand", "symbolSize": 10.638097333333334, "x": 162.63559, "y": 117.6565, "value": 20.457146, "category": 6}, {"id": "52", "name": "MmePontmercy", "symbolSize": 4.495239333333333, "x": 353.66415, "y": -205.89165, "value": 6.742859, "category": 6}, {"id": "53", "name": "MlleVaubois", "symbolSize": 2.6666666666666665, "x": 165.43939, "y": 339.7736, "value": 4, "category": 6}, {"id": "54", "name": "LtGillenormand", "symbolSize": 8.152382000000001, "x": 137.69348, "y": 196.1069, "value": 12.228573, "category": 6}, {"id": "55", "name": "Marius", "symbolSize": 12.58095333333333, "x": 206.44687, "y": -13.805411, "value": 53.37143, "category": 6}, {"id": "56", "name": "BaronessT", "symbolSize": 4.495239333333333, "x": 194.82993, "y": 224.78036, "value": 6.742859, "category": 6}, {"id": "57", "name": "Mabeuf", "symbolSize": 10.95238266666667, "x": 597.6618, "y": 135.18481, "value": 31.428574, "category": 8}, {"id": "58", "name": "Enjolras", "symbolSize": 12.266666666666666, "x": 355.78366, "y": -74.882454, "value": 42.4, "category": 8}, {"id": "59", "name": "Combeferre", "symbolSize": 10.95238266666667, "x": 515.2961, "y": -46.167564, "value": 31.428574, "category": 8}, {"id": "60", "name": "Prouvaire", "symbolSize": 11.295237333333333, "x": 614.29285, "y": -69.3104, "value": 25.942856, "category": 8}, {"id": "61", "name": "Feuilly", "symbolSize": 10.95238266666667, "x": 550.1917, "y": -128.17537, "value": 31.428574, "category": 8}, {"id": "62", "name": "Courfeyrac", "symbolSize": 10.609526666666667, "x": 436.17184, "y": -12.7286825, "value": 36.91429, "category": 8}, {"id": "63", "name": "Bahorel", "symbolSize": 10.780953333333333, "x": 602.55225, "y": 16.421427, "value": 34.17143, "category": 8}, {"id": "64", "name": "Bossuet", "symbolSize": 10.609526666666667, "x": 455.81955, "y": -115.45826, "value": 36.91429, "category": 8}, {"id": "65", "name": "Joly", "symbolSize": 10.780953333333333, "x": 516.40784, "y": 47.242233, "value": 34.17143, "category": 8}, {"id": "66", "name": "Grantaire", "symbolSize": 10.12381, "x": 646.4313, "y": -151.06331, "value": 28.685715, "category": 8}, {"id": "67", "name": "MotherPlutarch", "symbolSize": 2.6666666666666665, "x": 668.9568, "y": 204.65488, "value": 4, "category": 8}, {"id": "68", "name": "Gueulemer", "symbolSize": 10.12381, "x": 78.4799, "y": -347.15146, "value": 28.685715, "category": 7}, {"id": "69", "name": "Babet", "symbolSize": 10.12381, "x": 150.35959, "y": -298.50797, "value": 28.685715, "category": 7}, {"id": "70", "name": "Claquesous", "symbolSize": 10.12381, "x": 137.3717, "y": -410.2809, "value": 28.685715, "category": 7}, {"id": "71", "name": "Montparnasse", "symbolSize": 9.295237333333333, "x": 234.87747, "y": -400.85983, "value": 25.942856, "category": 7}, {"id": "72", "name": "Toussaint", "symbolSize": 6.323809333333333, "x": 40.942253, "y": 113.78272, "value": 9.485714, "category": 1}, {"id": "73", "name": "Child1", "symbolSize": 4.495239333333333, "x": 437.939, "y": 291.58234, "value": 6.742859, "category": 8},             {"id": "74", "name": "Child2", "symbolSize": 4.495239333333333, "x": 466.04922, "y": 283.3606, "value": 6.742859, "category": 8}, {"id": "75", "name": "Brujon", "symbolSize": 9.638097333333334, "x": 238.79364, "y": -314.06345, "value": 20.457146, "category": 7}, {"id": "76", "name": "MmeHucheloup", "symbolSize": 9.638097333333334, "x": 712.18353, "y": 4.8131495, "value": 20.457146, "category": 8}], "links": [{"source": "1", "target": "0"}, {"source": "2", "target": "0"}, {"source": "3", "target": "0"}, {"source": "3", "target": "2"}, {"source": "4", "target": "0"}, {"source": "5", "target": "0"}, {"source": "6", "target": "0"}, {"source": "7", "target": "0"}, {"source": "8", "target": "0"}, {"source": "9", "target": "0"}, {"source": "11", "target": "0"}, {"source": "11", "target": "2"}, {"source": "11", "target": "3"}, {"source": "11", "target": "10"}, {"source": "12", "target": "11"}, {"source": "13", "target": "11"}, {"source": "14", "target": "11"}, {"source": "15", "target": "11"}, {"source": "17", "target": "16"}, {"source": "18", "target": "16"}, {"source": "18", "target": "17"}, {"source": "19", "target": "16"}, {"source": "19", "target": "17"}, {"source": "19", "target": "18"}, {"source": "20", "target": "16"}, {"source": "20", "target": "17"}, {"source": "20", "target": "18"}, {"source": "20", "target": "19"}, {"source": "21", "target": "16"}, {"source": "21", "target": "17"}, {"source": "21", "target": "18"}, {"source": "21", "target": "19"}, {"source": "21", "target": "20"}, {"source": "22", "target": "16"}, {"source": "22", "target": "17"}, {"source": "22", "target": "18"}, {"source": "22", "target": "19"}, {"source": "22", "target": "20"}, {"source": "22", "target": "21"}, {"source": "23", "target": "11"}, {"source": "23", "target": "12"}, {"source": "23", "target": "16"}, {"source": "23", "target": "17"}, {"source": "23", "target": "18"}, {"source": "23", "target": "19"}, {"source": "23", "target": "20"}, {"source": "23", "target": "21"}, {"source": "23", "target": "22"}, {"source": "24", "target": "11"}, {"source": "24", "target": "23"}, {"source": "25", "target": "11"}, {"source": "25", "target": "23"}, {"source": "25", "target": "24"}, {"source": "26", "target": "11"}, {"source": "26", "target": "16"}, {"source": "26", "target": "24"}, {"source": "26", "target": "25"}, {"source": "27", "target": "11"}, {"source": "27", "target": "23"}, {"source": "27", "target": "24"}, {"source": "27", "target": "25"}, {"source": "27", "target": "26"}, {"source": "28", "target": "11"}, {"source": "28", "target": "27"}, {"source": "29", "target": "11"}, {"source": "29", "target": "23"}, {"source": "29", "target": "27"}, {"source": "30", "target": "23"}, {"source": "31", "target": "11"}, {"source": "31", "target": "23"}, {"source": "31", "target": "27"}, {"source": "31", "target": "30"}, {"source": "32", "target": "11"}, {"source": "33", "target": "11"}, {"source": "33", "target": "27"}, {"source": "34", "target": "11"}, {"source": "34", "target": "29"}, {"source": "35", "target": "11"}, {"source": "35", "target": "29"}, {"source": "35", "target": "34"}, {"source": "36", "target": "11"}, {"source": "36", "target": "29"}, {"source": "36", "target": "34"}, {"source": "36", "target": "35"}, {"source": "37", "target": "11"}, {"source": "37", "target": "29"}, {"source": "37", "target": "34"}, {"source": "37", "target": "35"}, {"source": "37", "target": "36"}, {"source": "38", "target": "11"}, {"source": "38", "target": "29"}, {"source": "38", "target": "34"}, {"source": "38", "target": "35"}, {"source": "38", "target": "36"}, {"source": "38", "target": "37"}, {"source": "39", "target": "25"}, {"source": "40", "target": "25"}, {"source": "41", "target": "24"}, {"source": "41", "target": "25"}, {"source": "42", "target": "24"}, {"source": "42", "target": "25"}, {"source": "42", "target": "41"}, {"source": "43", "target": "11"}, {"source": "43", "target": "26"}, {"source": "43", "target": "27"}, {"source": "44", "target": "11"}, {"source": "44", "target": "28"}, {"source": "45", "target": "28"}, {"source": "47", "target": "46"}, {"source": "48", "target": "11"}, {"source": "48", "target": "25"}, {"source": "48", "target": "27"}, {"source": "48", "target": "47"}, {"source": "49", "target": "11"}, {"source": "49", "target": "26"}, {"source": "50", "target": "24"}, {"source": "50", "target": "49"}, {"source": "51", "target": "11"}, {"source": "51", "target": "26"}, {"source": "51", "target": "49"}, {"source": "52", "target": "39"}, {"source": "52", "target": "51"}, {"source": "53", "target": "51"}, {"source": "54", "target": "26"}, {"source": "54", "target": "49"}, {"source": "54", "target": "51"}, {"source": "55", "target": "11"}, {"source": "55", "target": "16"}, {"source": "55", "target": "25"}, {"source": "55", "target": "26"}, {"source": "55", "target": "39"}, {"source": "55", "target": "41"}, {"source": "55", "target": "48"}, {"source": "55", "target": "49"}, {"source": "55", "target": "51"}, {"source": "55", "target": "54"}, {"source": "56", "target": "49"}, {"source": "56", "target": "55"}, {"source": "57", "target": "41"}, {"source": "57", "target": "48"}, {"source": "57", "target": "55"}, {"source": "58", "target": "11"}, {"source": "58", "target": "27"}, {"source": "58", "target": "48"}, {"source": "58", "target": "55"}, {"source": "58", "target": "57"}, {"source": "59", "target": "48"}, {"source": "59", "target": "55"}, {"source": "59", "target": "57"}, {"source": "59", "target": "58"}, {"source": "60", "target": "48"}, {"source": "60", "target": "58"}, {"source": "60", "target": "59"}, {"source": "61", "target": "48"}, {"source": "61", "target": "55"}, {"source": "61", "target": "57"}, {"source": "61", "target": "58"}, {"source": "61", "target": "59"}, {"source": "61", "target": "60"}, {"source": "62", "target": "41"}, {"source": "62", "target": "48"}, {"source": "62", "target": "55"}, {"source": "62", "target": "57"}, {"source": "62", "target": "58"}, {"source": "62", "target": "59"}, {"source": "62", "target": "60"}, {"source": "62", "target": "61"}, {"source": "63", "target": "48"}, {"source": "63", "target": "55"}, {"source": "63", "target": "57"}, {"source": "63", "target": "58"}, {"source": "63", "target": "59"}, {"source": "63", "target": "60"}, {"source": "63", "target": "61"}, {"source": "63", "target": "62"}, {"source": "64", "target": "11"}, {"source": "64", "target": "48"}, {"source": "64", "target": "55"}, {"source": "64", "target": "57"}, {"source": "64", "target": "58"}, {"source": "64", "target": "59"}, {"source": "64", "target": "60"}, {"source": "64", "target": "61"}, {"source": "64", "target": "62"}, {"source": "64", "target": "63"}, {"source": "65", "target": "48"}, {"source": "65", "target": "55"}, {"source": "65", "target": "57"}, {"source": "65", "target": "58"}, {"source": "65", "target": "59"}, {"source": "65", "target": "60"}, {"source": "65", "target": "61"}, {"source": "65", "target": "62"}, {"source": "65", "target": "63"}, {"source": "65", "target": "64"}, {"source": "66", "target": "48"}, {"source": "66", "target": "58"}, {"source": "66", "target": "59"}, {"source": "66", "target": "60"}, {"source": "66", "target": "61"}, {"source": "66", "target": "62"}, {"source": "66", "target": "63"}, {"source": "66", "target": "64"}, {"source": "66", "target": "65"}, {"source": "67", "target": "57"}, {"source": "68", "target": "11"}, {"source": "68", "target": "24"}, {"source": "68", "target": "25"}, {"source": "68", "target": "27"}, {"source": "68", "target": "41"}, {"source": "68", "target": "48"}, {"source": "69", "target": "11"}, {"source": "69", "target": "24"}, {"source": "69", "target": "25"}, {"source": "69", "target": "27"}, {"source": "69", "target": "41"}, {"source": "69", "target": "48"}, {"source": "69", "target": "68"}, {"source": "70", "target": "11"}, {"source": "70", "target": "24"}, {"source": "70", "target": "25"}, {"source": "70", "target": "27"}, {"source": "70", "target": "41"}, {"source": "70", "target": "58"}, {"source": "70", "target": "68"}, {"source": "70", "target": "69"}, {"source": "71", "target": "11"}, {"source": "71", "target": "25"}, {"source": "71", "target": "27"}, {"source": "71", "target": "41"}, {"source": "71", "target": "48"}, {"source": "71", "target": "68"}, {"source": "71", "target": "69"}, {"source": "71", "target": "70"}, {"source": "72", "target": "11"}, {"source": "72", "target": "26"}, {"source": "72", "target": "27"}, {"source": "73", "target": "48"}, {"source": "74", "target": "48"}, {"source": "74", "target": "73"}, {"source": "75", "target": "25"}, {"source": "75", "target": "41"}, {"source": "75", "target": "48"}, {"source": "75", "target": "68"}, {"source": "75", "target": "69"}, {"source": "75", "target": "70"}, {"source": "75", "target": "71"}, {"source": "76", "target": "48"}, {"source": "76", "target": "58"}, {"source": "76", "target": "62"}, {"source": "76", "target": "63"}, {"source": "76", "target": "64"}, {"source": "76", "target": "65"}, {"source": "76", "target": "66"}], "categories": [{"name": "0"}, {"name": "1"}, {"name": "2"}, {"name": "3"}, {"name": "4"}, {"name": "5"}, {"name": "6"}, {"name": "7"}, {"name": "8"}], "roam": true, "label": {"show": true, "position": "right", "formatter": "{b}"}, "labelLayout": {"hideOverlap": true}, "scaleLimit": {"min": 0.1, "max": 2}, "lineStyle": {"color": "source", "curveness": 0.3}}]}"""

    availabePlots = [
        {'data': plotA, 'de': 'Loss Kurve', 'en': 'Loss Curve'},
        {'data': plotB, 'de': 'Wetter', 'en': 'Weather'},
        {'data': plotC, 'de': 'Segmentierte Regionen', 'en': 'Segmented Regions'},
        {'data': plotD, 'de': 'Cluster', 'en': 'Cluster'},
        {'data': plotE, 'de': 'Großer Graph', 'en': 'Large Graph'},
    ]

    # Build query to generate 90 days of daily data entries for each location from (today-30) days to (today+60) days
    dailyDataQuery = """        
            INSERT INTO `daily_data`
            (id, location_id, date)
            VALUES """
    dt = date.today()  - timedelta(30)
    c = 1
    locs = ['uka', 'ukau', 'cha', 'ukbo', 'ukb', 'ukdd', 'ukd', 'fau', 'ume', 'ukf', 'ukfr', 'ukgi', 'umgö', 'ukgw', 'ukhal', 'uke', 'mhh', 'ukhd', 'ukhom', 'ukj', 'ukki', 'ukk', 'ukl', 'uklü', 'ummd', 'ukmz', 'ukma', 'ukmar', 'lmu', 'tum', 'ukm', 'ukr', 'umr', 'ukt', 'ukul', 'ukwü']
    for i in range(90):
        for loc in locs:
            line = f"({c}, '{loc}', '{dt.strftime('%Y-%m-%d')}'),\n"
            dailyDataQuery += line
            c += 1
        dt =  dt + timedelta(1)
    dailyDataQuery = dailyDataQuery[:-2] + ";"

    # Build query for random generation of measures and plot subsets for each of the previously created daily data entries
    plotDataQuery = """INSERT INTO `plot_data`
            (public_visible, name_de, name_en, plot_data, daily_data_id)
            VALUES 
            """
    measureDataQuery = """        
            INSERT INTO `measure_data`
            (value, daily_data_id, measure_id)
            VALUES 
            """
    dt = date.today()  - timedelta(30)
    dd_c = 1
    locs = ['uka', 'ukau', 'cha', 'ukbo', 'ukb', 'ukdd', 'ukd', 'fau', 'ume', 'ukf', 'ukfr', 'ukgi', 'umgö', 'ukgw', 'ukhal', 'uke', 'mhh', 'ukhd', 'ukhom', 'ukj', 'ukki', 'ukk', 'ukl', 'uklü', 'ummd', 'ukmz', 'ukma', 'ukmar', 'lmu', 'tum', 'ukm', 'ukr', 'umr', 'ukt', 'ukul', 'ukwü']
    locs_dict = {k: 0 for k in locs}
    measures = ['count_all', 'count_covid', 'quality_image', 'quality_noise', 'segmentation_sanity', 'slice_thickness', 'breathing_artifacts', 'trained_network']
    for i in range(90):
        for loc in locs:
            locs_dict[loc] += random.randint(0, 20)
            measureDataQuery += f"('{locs_dict[loc]}', {dd_c}, 'count_all'),\n"
            measureDataQuery += f"('{max(round(locs_dict[loc]/2) - random.randint(0, 20), 0)}', {dd_c}, 'count_covid'),\n"
            measureDataQuery += f"('{random.uniform(0,10)}', {dd_c}, 'quality_image'),\n"
            measureDataQuery += f"('{random.uniform(0,15)}', {dd_c}, 'quality_noise'),\n"
            measureDataQuery += f"('{random.uniform(0,100)}', {dd_c}, 'segmentation_sanity'),\n"
            measureDataQuery += f"('{random.randint(0,200)}', {dd_c}, 'slice_thickness'),\n"
            measureDataQuery += f"('{random.uniform(0,5)}', {dd_c}, 'breathing_artifacts'),\n"
            measureDataQuery += f"('{random.randint(0,1)}', {dd_c}, 'trained_network'),\n"

            for plot in random.sample(availabePlots, random.randint(0,len(availabePlots))):
                plotDataQuery += f"({random.randint(0,1)}, '{plot['de']}', '{plot['en']}', '{plot['data']}', {dd_c}),\n"

            dd_c += 1
        dt =  dt + timedelta(1)
    measureDataQuery = measureDataQuery[:-2] + ";"
    plotDataQuery = plotDataQuery[:-2] + ";"

    racoon_nodes = {
        "uka": "('uka', 50.77687044, 6.04336344, 'Aachen', 'Aachen', 'Klinik für Interventionelle und Diagnostische Radiologie, Uniklinik Aachen', 'Department of Interventional and Diagnostic Radiology, University Hospital Aachen')",
        "ukau": "('ukau', 48.38473826, 10.83822860, 'Augsburg', 'Augsburg', 'Klinik für Diagnostische und Interventionelle Radiologie und Neuroradiologie, Universitätsklinikum Augsburg', 'Department of Diagnostic and Interventional Radiology and Neuroradiology, University Hospital Augsburg')",
        "cha": "('cha', 52.52421379, 13.37826490, 'Berlin', 'Berlin', 'Diagnostische und interventionelle Radiologie und Nuklearmedizin, Charite - Universitätsmedizin Berlin', 'Diagnostic and Interventional Radiology and Nuclear Medicine, Charite - Universitätsmedizin Berlin')",
        "ukbo": "('ukbo', 51.46425904, 7.32764811, 'Bochum', 'Bochum', 'Diagnostische und Interventionelle Radiologie, Universitätsklinikum Bochum', 'Diagnostic and Interventional Radiology, University Hospital Bochum')",
        "ukb": "('ukb', 50.69864860, 7.10458293, 'Bonn', 'Bonn', 'Klinik für Interventionelle und Diagnostische Radiologie, Universitätsklinikum Bonn', 'Department of Interventional and Diagnostic Radiology, University Hospital Bonn')",
        "ukdd": "('ukdd', 51.05457243, 13.77649804, 'Dresden', 'Dresden', 'Institut und Poliklinik für Diagnostische und Interventionelle Radiologie, Universitätsklinikum Dresden', 'Institute and Polyclinic for Diagnostic and Interventional Radiology, University Hospital Dresden')",
        "ukd": "('ukd', 51.19487959, 6.79162287, 'Düsseldorf', 'Düsseldorf', 'Institut für Diagnostische und Interventionelle Radiologie, Universitätsklinikum Düsseldorf', 'Institute of Diagnostic and Interventional Radiology, University Hospital Düsseldorf')",
        "fau": "('fau', 49.60098237, 11.00954110, 'Erlangen', 'Erlangen', 'Radiologisches Institut, Universitätsklinikum Erlangen', 'Institute of Radiology, University Hospital Erlangen')",
        "ume": "('ume', 51.43660872, 6.98899359, 'Essen', 'Essen', 'Institut für Diagnostische und Interventionelle Radiologie und Neuroradiologie, Universitätsklinikum Essen', 'Institute of Diagnostic and Interventional Radiology and Neuroradiology, University Hospital Essen')",
        "ukf": "('ukf', 50.09570319, 8.66219894, 'Frankfurt', 'Frankfurt', 'Institut für Diagnostische und Interventionelle Radiologie, Universitätsklinikum Frankfurt', 'Institute of Diagnostic and Interventional Radiology, University Hospital Frankfurt')",
        "ukfr": "('ukfr', 48.00662459, 7.83916806, 'Freiburg', 'Freiburg', 'Klinik für Diagnostische und Interventionelle Radiologie, Universitätsklinikum Freiburg', 'Department of Radiology, University Hospital Freiburg')",
        "ukgi": "('ukgi', 50.57433510, 8.66410996, 'Gießen', 'Gießen', 'Klinik für Diagnostische und Interventionelle Radiologie, Universitätsklinikum Gießen', 'Department of Interventional and Diagnostic Radiology, University Hospital Gießen')",
        "umgö": "('umgö', 51.55108778, 9.94278207, 'Göttingen', 'Göttingen', 'Institut für Diagnostische und Interventionelle Radiologie, Universitätsklinikum Göttingen', 'Institute of Diagnostic and Interventional Radiology, University Hospital Göttingen')",
        "ukgw": "('ukgw', 54.08805474, 13.40513338, 'Greifswald', 'Greifswald', 'Zentrum für Radiologie, Universitätsklinikum Greifswald', 'Center for Radiology, University Hospital Greifswald')",
        "ukhal": "('ukhal', 51.50167347, 11.93742165, 'Halle', 'Halle', 'Universitätsklinik und Poliklinik für Radiologie, Universitätsklinikum Halle', 'Department of Radiology, University Hospital Halle')",
        "uke": "('uke', 53.59090704, 9.97397203, 'Hamburg', 'Hamburg', 'Diagnostische und Interventionelle Radiologie und Nuklearmedizin, Universitätsklinikum Hamburg', 'Diagnostic and Interventional Radiology and Nuclear Medicine, University Hospital Hamburg')",
        "mhh": "('mhh', 52.38277291, 9.80530370, 'Hannover', 'Hanover', 'Institut für Diagnostische und Interventionelle Radiologie, Universitätsklinikum Hannover', 'Institute of Diagnostic and Interventional Radiology, University Hospital Hanover')",
        "ukhd": "('ukhd', 49.42006772, 8.66750591, 'Heidelberg', 'Heidelberg', 'Klinik für Diagnostische und Interventionelle Radiologie, Universitätsklinikum Heidelberg und Deutsches Zentrum für Lungenforschung', 'Department of Diagnostic and Interventional Radiology, University Hospital Heidelberg and German Center for Lung Research')",
        "ukhom": "('ukhom', 49.30391003, 7.34802760, 'Homburg', 'Homburg', 'Klinik für Diagnostische und Interventionelle Radiologie, Universitätsklinikum Homburg', 'Department of Interventional and Diagnostic Radiology, University Hospital Homburg')",
        "ukj": "('ukj', 50.88491680, 11.62224410, 'Jena', 'Jena', 'Institut für Diagnostische und Interventionelle Radiologie, Universitätsklinikum Jena', 'Institute of Diagnostic and Interventional Radiology, University Hospital Jena')",
        "ukki": "('ukki', 54.32983192, 10.14069780, 'Kiel', 'Kiel', 'Klinik für Radiologie und Nuklearmedizin, Universitätsklinikum Schleswig-Holstein Kiel', 'Department of Radiology and Nuclear Medicine, University Hospital Schleswig-Holstein Kiel')",
        "ukk": "('ukk', 50.92415424, 6.91623605, 'Köln', 'Cologne', 'Institut für Diagnostische und Interventionelle Radiologie, Universitätsklinikum Köln', 'Institute of Diagnostic and Interventional Radiology, University Hospital Köln')",
        "ukl": "('ukl', 51.33138249, 12.38577654, 'Leipzig', 'Leipzig', 'Klinik und Poliklinik für Diagnostische und Interventionelle Radiologie, Universitätsklinikum Leipzig', 'Department of Diagnostic and Interventional Radiology, University Hospital Leipzig')",
        "uklü": "('uklü', 53.83427363, 10.70363872, 'Lübeck', 'Lübeck', 'Klinik für Radiologie und Nuklearmedizin, Universitätsklinikum Schleswig-Holstein, Lübeck', 'Department of Radiology and Nuclear Medicine, University Hospital Schleswig-Holstein, Lübeck')",
        "ummd": "('ummd', 52.10176414, 11.61899900, 'Magdeburg', 'Magdeburg', 'Universitätsklinik für Radiologie und Nuklearmedizin, Universitätsklinikum Magdeburg', 'Department of Radiology and Nuclear Medicine, University Hospital Magdeburg')",
        "ukmz": "('ukmz', 49.99424688, 8.25657289, 'Mainz', 'Mainz', 'Institut für Diagnostische und Interventionelle Radiologie, Universitätsklinikum Mainz', 'Institute of Diagnostic and Interventional Radiology, University Hospital Mainz')",
        "ukma": "('ukma', 49.49217534, 8.48566153, 'Mannheim', 'Mannheim', 'Klinik für Radiologie und Nuklearmedizin, Universitätsklinikum Mannheim', 'Department of Radiology and Nuclear Medicine, University Hospital Mannheim')",
        "ukmar": "('ukmar', 50.81430421, 8.80812882, 'Marburg', 'Marburg', 'Klinik für Diagnostische und Interventionelle Radiologie, Universitätsklinikum Marburg', 'Department of Interventional and Diagnostic Radiology, University Hospital Marburg')",
        "lmu": "('lmu', 48.11105832, 11.47119132, 'LMU München', 'LMU Munich', 'Klinik und Poliklinik für Radiologie, Universitätsklinikum der LMU München', 'Department of Radiology, University Hospital LMU Munich')",
        "tum": "('tum', 48.13706133, 11.59872312, 'TU München', 'TU Munich', 'Institut für diagnostische und interventionelle Radiologie, Universitätsklinikum der TU München', 'Institute of Diagnostic and Interventional Radiology, University Hospital TU Munich')",
        "ukm": "('ukm', 51.96058341, 7.59530538, 'Münster', 'Münster', 'Klinik für Radiologie, Universitätsklinikum Münster', 'Department of Radiology, University Hospital Münster')",
        "ukr": "('ukr', 48.98712489, 12.09067730, 'Regensburg', 'Regensburg', 'Institut für Röntgendiagnostik, Universitätsklinikum Regensburg', 'Department of Radiology, University Hospital Regensburg')",
        "umr": "('umr', 54.08585624, 12.10212174, 'Rostock', 'Rostock', 'Institut für Diagnostische und Interventionelle Radiologie, Universitätsklinikum Rostock', 'Institute of Diagnostic and Interventional Radiology, University Hospital Rostock')",
        "ukt": "('ukt', 48.52991697, 9.03750010, 'Tübingen', 'Tübingen', 'Abteilung für Diagnostische und Interventionelle Radiologie, Universitätsklinikum Tübingen', 'Department of Diagnostic and Interventional Radiology, University Hospital Tübingen')",
        "ukul": "('ukul', 48.42261781, 9.94882937, 'Ulm', 'Ulm', 'Klinik für Diagnostische und Interventionelle Radiologie, Universitätsklinikum Ulm', 'Department of Diagnostic and Interventional Radiology, University Hospital Ulm')",
        "ukwü": "('ukwü', 49.80412905, 9.95638180, 'Würzburg', 'Würzburg', 'Institut für Diagnostische und Interventionelle Radiologie, Universitätsklinikum Würzburg', 'Department of Diagnostic and Interventional Radiology, University Hospital Würzburg')"
    }
    node_id = os.getenv("RACOON_NODE_ID", "central")
    
    locations_insert_sql = """
        INSERT INTO `locations`
        (location_id, latitude, longitude, name_de, name_en, description_de, description_en)
        VALUES
    """
    if node_id == "central":
        locations_insert_sql += ",".join(racoon_nodes.values())+";"
    elif node_id in racoon_nodes:
        locations_insert_sql += racoon_nodes[node_id] + ";"
    else:
        raise Exception(f"Invalid racoon node id: {node_id}")

    # 1. Run query to create all locations
    operations = [

        migrations.RunSQL(locations_insert_sql),
        migrations.RunSQL("""
            INSERT INTO `measures`
            (measure_id, public_visible, is_main, is_color_default, is_size_default, is_open_ended, name_de, name_en, description_de, description_en, lower_bound, upper_bound)
            VALUES
            ('spike',1,1,0,0,1,'', '','spike', 'spike',0,0),
            ('resolution',1,1,0,0,1,'', '','resolution', 'resolution',0,0),
            ('noise',1,1,0,0,1,'', '','noise', 'noise',0,0),
            ('motion',1,1,0,0,1,'', '','motion', 'motion',0,0),
            ('ghosting',1,1,0,0,1,'', '','ghosting', 'ghosting',0,0),
            ('count_total',1,1,0,0,1,'', '','count_total', 'count_total',0,0),
            ('blur',1,1,0,0,1,'', '','blur', 'blur',0,0),
            ('LFC',1,1,0,0,1,'', '','LFC', 'LFC',0,0);
        """)

        # migrations.RunSQL("""
        #     INSERT INTO `measures`
        #     (measure_id, public_visible, is_main, is_color_default, is_size_default, is_open_ended, name_de, name_en, description_de, description_en, lower_bound, upper_bound)
        #     VALUES
        #     ('count_all',1,1,0,1,1,'Anzahl Gesamt','Number Total','Die Zahl aller Fälle','Number of cases',0,0),
        #     ('count_covid',0,1,0,0,1,'Anzahl Covid-19','Number Covid-19','Anzahl der Fälle mit nachgewiesener Covid-19 Infektion','Number of cases with confirmed Covid-19 infections',0,0),
        #     ('quality_noise',0,0,0,0,0,'Bildrauschen','Image Noise','Stärke des Bildrauschens','Amount of image noise',0,15),
        #     ('segmentation_sanity',0,1,1,0,0,'Segmentierungs-Plausibilität','Segmentation Sanity','Marker für die Qualität der Segmentierung. Prüft z.B. Anzahl und Größe der segmentierten Regionen','Marker for segmentation quality. Checks e.g. Count and size of segmented regions',0,100),
        #     ('quality_image',1,1,1,0,0,'Bildqualität','Image Quality','Automatisierter Marker zur Indikation der Bildqualität','Automated marker for indication of image quality',0,10),
        #     ('slice_thickness',1,0,0,0,1,'Schichtdicke','Slice Thickness','Durchschnittliche Schichtdicke aller Bilder','Averaged slice thickness of all images',0,0),
        #     ('breathing_artifacts',0,0,0,0,0,'Atemartefakte','Breathing Artifacts','Stärke der Atemartefakte','Amount of breathing artifacts',0,5),
        #     ('trained_network',0,1,0,0,0,'Netzwerk trainiert','Network trained','Gibt an, ob an diesem Tag ein KI-Netz trainiert wurde (0 = nein, 1 = ja)','Indicates if a AI-network was trained today (0 = no, 1 = yes)',0,1);
        # """),
        # # Run queries to generate the random data
        # migrations.RunSQL(dailyDataQuery),
        # migrations.RunSQL(measureDataQuery),
        # migrations.RunSQL(plotDataQuery),
    ]