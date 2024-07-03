import pickle
import argparse
from collections import namedtuple
import numpy as np
from shapely.geometry import Polygon


class DetectionIoUEvaluator(object):
    def __init__(self, iou_constraint=0.5, area_precision_constraint=0.5):
        self.iou_constraint = iou_constraint
        self.area_precision_constraint = area_precision_constraint
    def evaluate_image(self, gt, pred,value):
        def get_union(pD, pG):
            pD = Polygon(pD).buffer(0)
            pG = Polygon(pG).buffer(0)
            return pD.union(pG).area
        def get_intersection_over_union(pD, pG):
            iou = get_intersection(pD, pG) / get_union(pD, pG)
            return iou
        def get_intersection(pD, pG):
            pD = Polygon(pD).buffer(0)
            pG = Polygon(pG).buffer(0)
            return pD.intersection(pG).area
        def compute_ap(confList, matchList, numGtCare):
            correct = 0
            AP = 0
            if len(confList) > 0:
                confList = np.array(confList)
                matchList = np.array(matchList)
                sorted_ind = np.argsort(-confList)
                confList = confList[sorted_ind]
                matchList = matchList[sorted_ind]
                for n in range(len(confList)):
                    match = matchList[n]
                    if match:
                        correct += 1
                        AP += float(correct) / (n + 1)
                if numGtCare > 0:
                    AP /= numGtCare
            return AP

        perSampleMetrics = {}

        matchedSum = 0

        Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

        numGlobalCareGt = 0
        numGlobalCareDet = 0

        arrGlobalConfidences = []
        arrGlobalMatches = []

        recall = 0
        precision = 0
        hmean = 0
        E2E_recall = 0
        E2E_precision = 0
        E2E_hmean = 0
        detMatched = 0
        iouMat = np.empty([1, 1])
        E2E_Matched = 0
        gtPols = []
        detPols = []

        gtPolPoints = []
        detPolPoints = []

        # Array of Ground Truth Polygons' keys marked as don't Care
        gtDontCarePolsNum = []
        # Array of Detected Polygons' matched with a don't Care GT
        detDontCarePolsNum = []

        pairs = []
        detMatchedNums = []

        arrSampleConfidences = []
        arrSampleMatch = []

        evaluationLog = ""

        for n in range(len(gt)):
            points = gt[n]['points']
            # transcription = gt[n]['text']
            dontCare = gt[n]['ignore']

            # if not Polygon(points).is_valid or not Polygon(points).is_simple:
            if not Polygon(points).buffer(0).is_simple:
                continue

            gtPol = points
            gtPols.append(gtPol)
            gtPolPoints.append(points)
            if dontCare:
                gtDontCarePolsNum.append(len(gtPols) - 1)

        evaluationLog += "GT polygons: " + str(len(gtPols)) + (
            " (" + str(len(gtDontCarePolsNum)) +
            " don't care)\n" if len(gtDontCarePolsNum) > 0 else "\n")

        for n in range(len(pred)):
            points = pred[n]['points']

            # TODO 忽略
            if not Polygon(points).buffer(0).is_valid or \
                    not Polygon(points).buffer(0).is_simple:
                continue

            detPol = points
            detPols.append(detPol)
            detPolPoints.append(points)
            if len(gtDontCarePolsNum) > 0:
                for dontCarePol in gtDontCarePolsNum:
                    dontCarePol = gtPols[dontCarePol]
                    intersected_area = get_intersection(dontCarePol, detPol)
                    pdDimensions = Polygon(detPol).area
                    precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                    # TODO 标记为不重要
                    if (precision > self.area_precision_constraint):
                        detDontCarePolsNum.append(len(detPols) - 1)
                        break

        evaluationLog += "DET polygons: " + str(len(detPols)) + (
            " (" + str(len(detDontCarePolsNum)) +
            " don't care)\n" if len(detDontCarePolsNum) > 0 else "\n")


        if len(gtPols) > 0 and len(detPols) > 0:
            # Calculate IoU and precision matrixs
            outputShape = [len(gtPols), len(detPols)]
            iouMat = np.empty(outputShape)
            gtRectMat = np.zeros(len(gtPols), np.int8)
            detRectMat = np.zeros(len(detPols), np.int8)
            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    pG = gtPols[gtNum]
                    pD = detPols[detNum]
                    iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)

            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    # TODO 尚未被匹配并且不在忽略范围内
                    if gtRectMat[gtNum] == 0 and detRectMat[
                            detNum] == 0 and gtNum not in gtDontCarePolsNum and detNum not in detDontCarePolsNum:
                        # TODO 大于阈值
                        if iouMat[gtNum, detNum] > self.iou_constraint:
                            gtRectMat[gtNum] = 1
                            detRectMat[detNum] = 1
                            # TODO 检测匹配加1
                            detMatched += 1
                            if value[gtNum]:
                                E2E_Matched +=1
                            pairs.append({'gt': gtNum, 'det': detNum})
                            detMatchedNums.append(detNum)
                            evaluationLog += "Match GT #" + \
                                str(gtNum) + " with Det #" + str(detNum) + "\n"

        numGtCare = (len(gtPols) - len(gtDontCarePolsNum))
        numDetCare = (len(detPols) - len(detDontCarePolsNum))
        #TODO GT个数为0,则召回率为1,端到端的召回率也为0,如果检测数大于0,则精度为0,否则为1
        if numGtCare == 0:
            recall = float(1)
            E2E_recall = float(1)
            precision = float(0) if numDetCare > 0 else float(1)
            E2E_precision = float(0) if numDetCare > 0 else float(1)
        else:
            recall = float(detMatched) / numGtCare
            E2E_recall = float(E2E_Matched) / numGtCare
            precision = 0 if numDetCare == 0 else float(
                detMatched) / numDetCare
            E2E_precision  = 0 if numDetCare == 0 else float(
                E2E_Matched) / numDetCare
        hmean = 0 if (precision + recall) == 0 else 2.0 * \
            precision * recall / (precision + recall)
        E2E_hmean = 0 if (E2E_precision + E2E_recall) == 0 else 2.0 * \
            E2E_precision * E2E_recall / (E2E_precision + E2E_recall)
        matchedSum += detMatched
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare

        perSampleMetrics = {
            'precision': precision,
            'recall': recall,
            'hmean': hmean,
            'pairs': pairs,
            'iouMat': [] if len(detPols) > 100 else iouMat.tolist(),
            'gtPolPoints': gtPolPoints,
            'detPolPoints': detPolPoints,
            'gtCare': numGtCare,
            'detCare': numDetCare,
            'gtDontCare': gtDontCarePolsNum,
            'detDontCare': detDontCarePolsNum,
            'detMatched': detMatched,
            'evaluationLog': evaluationLog,
            'E2E_precision': E2E_precision,
            'E2E_recall': E2E_recall,
            'E2E_hmean': E2E_hmean,
            'E2E_Matched':E2E_Matched
        }

        return perSampleMetrics

    def combine_results(self, results):
        numGlobalCareGt = 0
        numGlobalCareDet = 0
        matchedSum = 0
        matchedSum_E2E = 0
        for result in results:
            numGlobalCareGt += result['gtCare']
            numGlobalCareDet += result['detCare']
            matchedSum += result['detMatched']
            matchedSum_E2E+=result['E2E_Matched']
        methodRecall = 0 if numGlobalCareGt == 0 else float(
            matchedSum) / numGlobalCareGt
        methodPrecision = 0 if numGlobalCareDet == 0 else float(
            matchedSum) / numGlobalCareDet
        methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * \
            methodRecall * methodPrecision / (methodRecall + methodPrecision)

        # TODO E2E
        E2E_methodRecall = 0 if numGlobalCareGt == 0 else float(
            matchedSum_E2E) / numGlobalCareGt
        E2E_methodPrecision = 0 if numGlobalCareDet == 0 else float(
            matchedSum_E2E) / numGlobalCareDet
        E2E_methodHmean = 0 if E2E_methodRecall + E2E_methodPrecision == 0 else 2 * \
            E2E_methodRecall * methodPrecision / (E2E_methodRecall + E2E_methodPrecision)
        methodMetrics = {
            'precision': methodPrecision,
            'recall': methodRecall,
            'hmean': methodHmean,
            'E2E_precision':E2E_methodPrecision,
            'E2E_recall':E2E_methodRecall,
            'E2E_hmean':E2E_methodHmean
        }

        return methodMetrics
# model = DetectionIoUEvaluator()
# import pickle
# with open('my_list_gt.pkl', 'rb') as f:
#     gt = pickle.load(f)
# with open('my_list_pred.pkl', 'rb') as f:
#     pred = pickle.load(f)
# model.evaluate_image(gt,pred)
def load_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--area', type=float, default=0.5)
    parser.add_argument('--poly_gts_fp',
                        type=str,
                        default='./data/result_poly_gts.pkl')
    parser.add_argument('--poly_preds_fp',
                        type=str,
                        default='./data/result_poly_preds.pkl')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = load_args()
    evaluator = DetectionIoUEvaluator(iou_constraint=args.iou,
                                      area_precision_constraint=args.area)

    # # pseudo code
    # preds = []
    # for img in imgs:
    #     for text_pred in img:
    #         pred_sample = {
    #             # polygon, list of point coordinates
    #             "points": [(0, 0), (1, 0), (1, 1), (0, 1)],
    #             "text": "aaa",
    #             "ignore": False
    #         }

    # gts = [
    #     [
    #         {
    #             'points': [
    #                 (0, 0), (1, 0), (1, 1), (0, 1),
    #                 # (-1, 1),
    #             ],
    #             'text': 1234,
    #             'ignore': False,
    #         },
    #         {
    #             'points': [(2, 2), (3, 2), (3, 3), (2, 3)],
    #             'text': 5678,
    #             'ignore': True,
    #         }
    #     ]
    # ]
    # preds = [[
    #     {
    #         'points': [
    #             (0.1, 0.1), (1, 0), (1, 1), (0, 1)
    #         ],
    #         'text': 123,
    #         'ignore': False,
    #     }
    # ]]

    with open(args.poly_gts_fp, "rb") as f:
        gts = pickle.load(f)

    with open(args.poly_preds_fp, "rb") as f:
        preds = pickle.load(f)

    results = []
    # for each images
    for gt, pred in zip(gts, preds):
        results.append(evaluator.evaluate_image(gt, pred))
    metrics = evaluator.combine_results(results)
    print(metrics)
