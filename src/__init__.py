"""
FWI Pipeline — Полный цикл Full Waveform Inversion.

Модули:
- data:       загрузка SEG-Y + предобработка
- models:     архитектуры MLP и PINN
- training:   циклы обучения с early stopping
- evaluation: расчёт метрик
- visualization: построение графиков
- utils:      логирование
"""
