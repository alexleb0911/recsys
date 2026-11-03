# Рекомендательные системы

Студент: **Лебедев Александр Владимирович**

## Домашнее задание 1

Сравнение ItemKNN, iALS, EASE, FM и iALS с сигналом из частоты и давности покупок на dunnhumby The Complete Journey.

Основная работа: [hw1/experiments.ipynb](hw1/experiments.ipynb). Описание данных: [hw1/DATA.md](hw1/DATA.md).

Запуск с Python 3.11 или 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r hw1/requirements.txt
cd hw1
jupyter lab experiments.ipynb
```

В Jupyter выполнить Restart Kernel and Run All Cells. При первом запуске код попробует скачать данные в пользовательский кеш. Загрузка зависит от доступности Kaggle; исходные данные в репозиторий не входят. В ноутбуке приведены источник, правила отбора и все результаты.

Работа размещается в папке `hw1` в ветке `master`.

## Если загрузка данных недоступна

Скачать архив [The Complete Journey, CSV release 2014, версия 1](https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey/versions/1) и распаковать `transaction_data.csv` и `product.csv` в одну папку. Перед запуском Jupyter задать путь:

```bash
export DUNNHUMBY_DIR="/полный/путь/к/папке"
cd hw1
jupyter lab experiments.ipynb
```

Команды выше рассчитаны на macOS/Linux. В PowerShell переменная задаётся как `$env:DUNNHUMBY_DIR = "C:\путь\к\папке"`. Проверяются SHA-256 исходных CSV. Другая версия данных не подставляется автоматически.
