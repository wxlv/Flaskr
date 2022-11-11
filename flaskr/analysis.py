from . import deal_process2
from flask import Blueprint, flash, g, redirect, request, json

bp = Blueprint("analysis", __name__, url_prefix="/analysis")


@bp.route("/single", methods=["POST"])
def singleAnalysis():
    reqdata = request.get_json()
    print(reqdata["SourceData"])
    deal_process2.run_main()
    result = deal_process2.describe_analysis([reqdata["SourceData"]])
    # deal_process2.single_analysis()
    return result
    data = request.form["sourcedata"]
    col1 = request.form["groupcol"]
    col2 = request.form["obsercol"]
    rst = single_analysis(data, col1, col2)
