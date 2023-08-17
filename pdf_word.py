# -*- coding: utf-8 -*-
import os
import shutil
from io import BytesIO
from flask import Flask, request, send_file
from flask_cors import CORS
from pdf2docx import Converter
from uuid import uuid4
import json
import traceback
from docx2pdf import convert
import pythoncom


def convert1(pdf_file, docx_file):
    cv = Converter(pdf_file)
    cv.convert(docx_file, start=0, end=None)
    cv.close()

app = Flask(__name__)
CORS(app, supports_credentials=True)


@app.route("/pdf2word",methods=["POST"])
def pdf2word():
    """
    pdf文件转word文件
    """
    try:
        cache_path = os.path.join(os.getcwd(), uuid4().hex)
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        file = request.files['file']
        file_name = file.filename
        file.save(os.path.join(cache_path,file_name))
        convert1(os.path.join(cache_path,file_name),os.path.join(cache_path,file_name+".docx"))
        read_doc = open(os.path.join(cache_path,file_name+".docx"), 'rb')
        f = read_doc.read()
        read_doc.close()
        f = BytesIO(f)
        # return json.dump({"file":f})
        return send_file(f,download_name=file_name.split(".")[0] + ".docx")
    except:
        return json.dumps({"code": 999, "msg": traceback.format_exc(), "result": {}})
    finally:
        shutil.rmtree(cache_path)


@app.route("/word2pdf",methods=["POST"])
def word2pdf():
    """
    word文件转pdf文件
    """
    try:
        pythoncom.CoInitialize()
        cache_path = os.path.join(os.getcwd(), uuid4().hex)
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        file = request.files['file']
        # file_stream = file.read()
        file_name = file.filename
        file.save(os.path.join(cache_path, file_name))
        convert(os.path.join(cache_path, file_name),os.path.join(cache_path, file_name+".pdf"))
        # cv = convert(os.path.join(cache_path, file_name),os.path.join(cache_path, file_name+".pdf"))
        read_doc = open(os.path.join(cache_path, file_name + ".pdf"), 'rb')
        f = read_doc.read()
        read_doc.close()
        f = BytesIO(f)
        # return json.dump({"file":f})
        return send_file(f, download_name= "out.pdf")

    except:
        print(traceback.print_exc())
        return json.dumps({"code": 999, "msg": traceback.format_exc(), "result": {}})
    finally:
        shutil.rmtree(cache_path)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8889, debug=False, threaded=True)
