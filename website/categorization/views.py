import os

from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader, Context, Template
from django.template.loader import render_to_string
from django import forms

from make_prediction import *
from build_metadata import *
from utils.loading import *
from utils.paths import *

fileContent = ''
fileName = ''
predictions = []

def classification(request):
    global fileContent, fileName, predictions

    fileContent = request.FILES['file'].read()
    fileName = request.FILES['file'].name

    current_path = os.getcwd()
    os.chdir('../multi_label_with_level/')

    data = LR(fileContent)
    results = get_tag_silence(predict_silence('LR', data))
    os.chdir(current_path)

    predictions = results
    return results

def rebuild_metadata():
    build_metadata_and_topic_dict(os.getcwd(), global_doc_topic_id, 0, metadata, topic_dict)

    json.dump(metadata, open('metadata.json', 'w'))
    json.dump(topic_dict, open('topic_dict.json', 'w'))

def saveFile():
    global predictions, fileContent, fileName

    current_path = os.getcwd()
    os.chdir('../training_data/techone_new_training_data/')
    print('Name:' + fileName)
    for prediction in predictions:
        with open(prediction['tag'] + '/' + fileName, 'wb') as file:
            file.write(fileContent)

    rebuild_metadata()
    os.chdir(current_path)

def download(filePath, fileName, request):
    with open(filePath + '/' + fileName, 'r') as file:
        response = HttpResponse(file.read(), content_type="application/force-download")
        response['Content-Disposition'] = 'attachment; filename="{}"'.format(fileName)

    return response


def index(request):
    results = None

    if 'file' in request.FILES:
        results = classification(request)

    id = 0
    if 'id' in request.GET:
        id = int(request.GET['id'])

    if 'save' in request.GET:
        saveFile()

    meta_data = load_metadata(Path.techone_training_data() + 'metadata.json')

    if 'dl' in request.GET:
        file_id = int(request.GET['dl'])
        filePath = '../training_data/techone_new_training_data/' + meta_data[file_id]['doc_topic'][0]
        return download(filePath, meta_data[file_id]['doc_name'], request)

    context = {
                'meta_data': meta_data,
                'results': results,
                'file_info': {
                                'name': meta_data[id]['doc_name'],
                                'date': meta_data[id]['doc_date'],
                                'tags': meta_data[id]['doc_topic']
                }}

    template = loader.get_template("categorization/index.html");
    return HttpResponse(template.render(context, request));