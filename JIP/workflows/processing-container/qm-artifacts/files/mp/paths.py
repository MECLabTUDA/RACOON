# ------------------------------------------------------------------------------
# Module where paths should be defined.
# ------------------------------------------------------------------------------
import os

# Path where intermediate and final results are stored
storage_path = '/gris/gris-f/homestud/aranem/medical_pytorch-storage'
storage_data_path = os.path.join(storage_path, 'data')

# Original data paths. TODO: set necessary data paths.
original_data_paths = {'DecathlonLeftAtrium': '/gris/gris-f/homestud/aranem/medical_pytorch-storage/data/Task02_Heart',
                       'DecathlonLung': '/gris/gris-f/homestud/aranem/medical_pytorch-storage/data/Task06_Lung',
                       'FRACorona': '/gris/gris-f/homestud/aranem/medical_pytorch-storage/data/UK_Frankfurt2',
                       'GC_Corona': '/gris/gris-f/homestud/aranem/medical_pytorch-storage/data/COVID-19-20_Grand_Challenge',
                       'RadiopediaTrain': '/gris/gris-f/homestud/aranem/medical_pytorch-storage/data/Covid-RACOON/All_images_and_labels/Task100_RadiopediaTrain',
                       'RadiopediaTest': '/gris/gris-f/homestud/aranem/medical_pytorch-storage/data/Covid-RACOON/All_images_and_labels/Task101_RadiopediaTest',
                       'MosmedTrain': '/gris/gris-f/homestud/aranem/medical_pytorch-storage/data/Covid-RACOON/All_images_and_labels/Task200_MosmedTrain',
                       'MosmedTest': '/gris/gris-f/homestud/aranem/medical_pytorch-storage/data/Covid-RACOON/All_images_and_labels/Task201_MosmedTest',
                       'FRACoronaTrain': '/gris/gris-f/homestud/aranem/medical_pytorch-storage/data/Covid-RACOON/All_images_and_labels/Task541_FrankfurtTrainF4',
                       'FRACoronaTest': '/gris/gris-f/homestud/aranem/medical_pytorch-storage/data/Covid-RACOON/All_images_and_labels/Task542_FrankfurtTestF4',
                       'GC_CoronaTrain': '/gris/gris-f/homestud/aranem/medical_pytorch-storage/data/Covid-RACOON/All_images_and_labels/Task740_ChallengeTrainF4',
                       'GC_CoronaTest': '/gris/gris-f/homestud/aranem/medical_pytorch-storage/data/Covid-RACOON/All_images_and_labels/Task741_ChallengeTestF4'}

# Path that represents JIP data structure for training and inference
JIP_dir = os.path.join(storage_path, 'JIP')
JIP_dir = os.path.join("/", os.environ["WORKFLOW_DIR"])

# Login for Telegram Bot
telegram_login = {'chat_id': '', 'token': ''}