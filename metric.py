import numpy as np
from scipy.ndimage import center_of_mass, convolve, distance_transform_edt as bwdist

#import matplotlib.pyplot as plt



class MetricRecorder:
    def __init__(self, eval_length_for_fm, beta_for_wfm=1):
        self.mae = cal_mae()
        self.fm = cal_fm(eval_length_for_fm)
        self.sm = cal_sm()
        self.em = cal_em()
        self.wfm = cal_wfm(beta_for_wfm)

    def update(self, pre, gt):
        self.mae.update(pre, gt)
        self.sm.update(pre, gt)
        self.fm.update(pre, gt)
        self.em.update(pre, gt)
        self.wfm.update(pre, gt)

    def show(self, bit_num: int) -> tuple:
        mae = self.mae.show()
        f, maxf, meanf, p, r = self.fm.show()
        sm = self.sm.show()
        em = self.em.show()
        wfm = self.wfm.show()
        mae, maxf, meanf, sm, em, wfm = self.round_bit_num(
            data=[mae, maxf, meanf, sm, em, wfm], bit_num=bit_num
        )
        return mae, (maxf, meanf, f, p, r), sm, em, wfm

    def round_bit_num(self, data: list, bit_num: int):
        results = []
        for d in data:
            results.append(round(d, bit_num))
        return results

'''
class CurveDrawer:
    def __init__(self, row_num, col_num):
        self.fig = plt.figure()
        self.row_num = row_num
        self.col_num = col_num
        self.font_cfg = {
            "title": {"fontsize": 12, "fontweight": "semibold", "fontname": "serif",},
            "label": {"fontsize": 10, "fontweight": "normal", "fontname": "serif",},
            "ticks": {"fontsize": 8, "fontweight": "normal", "fontname": "serif",},
            "legend": {"size": 8, "weight": "normal", "family": "serif",},
        }

    def add_subplot(self, curr_idx):
        assert isinstance(curr_idx, int) and curr_idx > 0
        self.ax = self.fig.add_subplot(self.row_num, self.col_num, curr_idx)
        self.ax.grid(True)

    def draw_method_curve(
        self,
        dataset_name,
        method_curve_setting,
        x_label,
        y_label,
        x_data,
        y_data,
        x_lim: tuple = (0, 1),
        y_lim: tuple = (0.1, 1),
    ):
        """
        :param method_curve_setting:  {
                "line_color": "color"(str),
                "line_style": "style"(str),
                "line_label": "label"(str),
                "line_width": width(int),
            }
        """
        # give plot a title
        self.ax.set_title(dataset_name, fontdict=self.font_cfg["title"])

        # make axis labels
        self.ax.set_xlabel(x_label, fontdict=self.font_cfg["label"])
        self.ax.set_ylabel(y_label, fontdict=self.font_cfg["label"])

        # 对坐标刻度的设置
        label = ["{.2f}".format(x) for x in np.linspace(0, 1, 11)]
        self.ax.set_xticks(np.linspace(0, 1, 11))
        self.ax.set_yticks(np.linspace(0, 1, 11))
        self.ax.set_xticklabels(labels=label, fontdict=self.font_cfg["ticks"])
        self.ax.set_yticklabels(labels=label, fontdict=self.font_cfg["ticks"])

        self.ax.set_xlim(x_lim)
        self.ax.set_ylim(y_lim)

        # [CPFP, "red", "-", "OURS", 3],
        self.ax.plot(
            x_data,
            y_data,
            linewidth=method_curve_setting["line_width"],
            label=method_curve_setting["line_label"],
            color=method_curve_setting["line_color"],
            linestyle=method_curve_setting["line_style"],
        )

        # loc=0，自动将位置放在最合适的
        self.ax.legend(loc=3, prop=self.font_cfg["legend"])

        # 对坐标轴的框线进行设置, 设置轴
        self.ax.spines["top"].set_linewidth(1)
        self.ax.spines["bottom"].set_linewidth(1)
        self.ax.spines["left"].set_linewidth(1)
        self.ax.spines["right"].set_linewidth(1)

    def show(self):
        plt.show()
'''

def normalize_pil(pre, gt):
    gt = np.asarray(gt)
    pre = np.asarray(pre)
    gt = gt / (gt.max() + 1e-8)
    gt = np.where(gt > 0.5, 1, 0)
    max_pre = pre.max()
    min_pre = pre.min()
    if max_pre == min_pre:
        pre = pre / 255
    else:
        pre = (pre - min_pre) / (max_pre - min_pre)
    return pre, gt

class cal_fm(object):
    # Fmeasure(maxFm, meanFm)---Frequency-tuned salient region detection(CVPR 2009)
    def __init__(self, num, thds=255):
        self.num = num
        self.thds = thds
        self.precision = np.zeros((self.num, self.thds))
        self.recall = np.zeros((self.num, self.thds))
        self.meanF = np.zeros((self.num, 1))
        self.idx = 0

    def update(self, pred, gt):
        if gt.max() != 0:
            prediction, recall, Fmeasure_temp = self.cal(pred, gt)
            self.precision[self.idx, :] = prediction
            self.recall[self.idx, :] = recall
            self.meanF[self.idx, :] = Fmeasure_temp
        self.idx += 1

    def cal(self, pred, gt):
        ########################meanF##############################
        th = 2 * pred.mean()
        if th > 1:
            th = 1
        binary = np.zeros_like(pred)
        binary[pred >= th] = 1
        hard_gt = np.zeros_like(gt)
        hard_gt[gt > 0.5] = 1
        tp = (binary * hard_gt).sum()
        if tp == 0:
            meanF = 0
        else:
            pre = tp / binary.sum()
            rec = tp / hard_gt.sum()
            meanF = 1.3 * pre * rec / (0.3 * pre + rec)

        ########################maxF##############################
        pred = np.uint8(pred * 255)
        target = pred[gt > 0.5]
        nontarget = pred[gt <= 0.5]
        targetHist, _ = np.histogram(target, bins=range(256))
        nontargetHist, _ = np.histogram(nontarget, bins=range(256))
        targetHist = np.cumsum(np.flip(targetHist, axis=0), axis=0)
        nontargetHist = np.cumsum(np.flip(nontargetHist, axis=0), axis=0)
        precision = targetHist / (targetHist + nontargetHist + 1e-8)
        recall = targetHist / np.sum(gt)
        return precision, recall, meanF

    def show(self):
        assert self.num == self.idx, "{}, {}".format(self.num, self.idx)
        precision = self.precision.mean(axis=0)
        recall = self.recall.mean(axis=0)
        fmeasure = 1.3 * precision * recall / (0.3 * precision + recall + 1e-8)
        fmeasure_avg = self.meanF.mean(axis=0)
        return fmeasure, fmeasure.max(), fmeasure_avg[0], precision, recall


class cal_mae(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def cal(self, pred, gt):
        return np.mean(np.abs(pred - gt))

    def show(self):
        return np.mean(self.prediction)


class cal_sm(object):
    # Structure-measure: A new way to evaluate foreground maps (ICCV 2017)
    def __init__(self, alpha=0.5):
        self.prediction = []
        self.alpha = alpha

    def update(self, pred, gt):
        gt = gt > 0.5
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def show(self):
        return np.mean(self.prediction)

    def cal(self, pred, gt):
        y = np.mean(gt)
        if y == 0:
            score = 1 - np.mean(pred)
        elif y == 1:
            score = np.mean(pred)
        else:
            score = self.alpha * self.object(pred, gt) + (1 - self.alpha) * self.region(
                pred, gt
            )
        return score

    def object(self, pred, gt):
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)

        u = np.mean(gt)
        return u * self.s_object(fg, gt) + (1 - u) * self.s_object(
            bg, np.logical_not(gt)
        )

    def s_object(self, in1, in2):
        x = np.mean(in1[in2])
        sigma_x = np.std(in1[in2])
        return 2 * x / (pow(x, 2) + 1 + sigma_x + 1e-8)

    def region(self, pred, gt):
        [y, x] = center_of_mass(gt)
        y = int(round(y)) + 1
        x = int(round(x)) + 1
        [gt1, gt2, gt3, gt4, w1, w2, w3, w4] = self.divideGT(gt, x, y)
        pred1, pred2, pred3, pred4 = self.dividePred(pred, x, y)

        score1 = self.ssim(pred1, gt1)
        score2 = self.ssim(pred2, gt2)
        score3 = self.ssim(pred3, gt3)
        score4 = self.ssim(pred4, gt4)

        return w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4

    def divideGT(self, gt, x, y):
        h, w = gt.shape
        area = h * w
        LT = gt[0:y, 0:x]
        RT = gt[0:y, x:w]
        LB = gt[y:h, 0:x]
        RB = gt[y:h, x:w]

        w1 = x * y / area
        w2 = y * (w - x) / area
        w3 = (h - y) * x / area
        w4 = (h - y) * (w - x) / area

        return LT, RT, LB, RB, w1, w2, w3, w4

    def dividePred(self, pred, x, y):
        h, w = pred.shape
        LT = pred[0:y, 0:x]
        RT = pred[0:y, x:w]
        LB = pred[y:h, 0:x]
        RB = pred[y:h, x:w]

        return LT, RT, LB, RB

    def ssim(self, in1, in2):
        in2 = np.float32(in2)
        h, w = in1.shape
        N = h * w

        x = np.mean(in1)
        y = np.mean(in2)
        sigma_x = np.var(in1)
        sigma_y = np.var(in2)
        sigma_xy = np.sum((in1 - x) * (in2 - y)) / (N - 1)

        alpha = 4 * x * y * sigma_xy
        beta = (x * x + y * y) * (sigma_x + sigma_y)

        if alpha != 0:
            score = alpha / (beta + 1e-8)
        elif alpha == 0 and beta == 0:
            score = 1
        else:
            score = 0

        return score


class cal_em(object):
    # Enhanced-alignment Measure for Binary Foreground Map Evaluation (IJCAI 2018)
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def cal(self, pred, gt):
        th = 2 * pred.mean()
        if th > 1:
            th = 1
        FM = np.zeros(gt.shape)
        FM[pred >= th] = 1
        FM = np.array(FM, dtype=bool)
        GT = np.array(gt, dtype=bool)
        dFM = np.double(FM)
        if sum(sum(np.double(GT))) == 0:
            enhanced_matrix = 1.0 - dFM
        elif sum(sum(np.double(~GT))) == 0:
            enhanced_matrix = dFM
        else:
            dGT = np.double(GT)
            align_matrix = self.AlignmentTerm(dFM, dGT)
            enhanced_matrix = self.EnhancedAlignmentTerm(align_matrix)
        [w, h] = np.shape(GT)
        score = sum(sum(enhanced_matrix)) / (w * h - 1 + 1e-8)
        return score

    def AlignmentTerm(self, dFM, dGT):
        mu_FM = np.mean(dFM)
        mu_GT = np.mean(dGT)
        align_FM = dFM - mu_FM
        align_GT = dGT - mu_GT
        align_Matrix = (
            2.0
            * (align_GT * align_FM)
            / (align_GT * align_GT + align_FM * align_FM + 1e-8)
        )
        return align_Matrix

    def EnhancedAlignmentTerm(self, align_Matrix):
        enhanced = np.power(align_Matrix + 1, 2) / 4
        return enhanced

    def show(self):
        return np.mean(self.prediction)


class cal_wfm(object):
    """
    created by lartpang (Youwei Pang)
    """

    def __init__(self, beta=1):
        self.beta = beta
        self.eps = 1e-6
        self.scores_list = []

    def update(self, pred, gt):
        assert pred.ndim == gt.ndim and pred.shape == gt.shape
        assert pred.max() <= 1 and pred.min() >= 0
        assert gt.max() <= 1 and gt.min() >= 0

        gt = gt > 0.5
        if gt.max() == 0:
            score = 0
        else:
            score = self.cal(pred, gt)
        self.scores_list.append(score)

    def matlab_style_gauss2D(self, shape=(7, 7), sigma=5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m, n = [(ss - 1.0) / 2.0 for ss in shape]
        y, x = np.ogrid[-m : m + 1, -n : n + 1]
        h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def cal(self, pred, gt):
        # [Dst,IDXT] = bwdist(dGT);
        Dst, Idxt = bwdist(gt == 0, return_indices=True)

        # %Pixel dependency
        # E = abs(FG-dGT);
        E = np.abs(pred - gt)
        # Et = E;
        # Et(~GT)=Et(IDXT(~GT)); %To deal correctly with the edges of the foreground region
        Et = np.copy(E)
        Et[gt == 0] = Et[Idxt[0][gt == 0], Idxt[1][gt == 0]]

        # K = fspecial('gaussian',7,5);
        # EA = imfilter(Et,K);
        # MIN_E_EA(GT & EA<E) = EA(GT & EA<E);
        K = self.matlab_style_gauss2D((7, 7), sigma=5)
        EA = convolve(Et, weights=K, mode="constant", cval=0)
        MIN_E_EA = np.where(gt & (EA < E), EA, E)

        # %Pixel importance
        # B = ones(size(GT));
        # B(~GT) = 2-1*exp(log(1-0.5)/5.*Dst(~GT));
        # Ew = MIN_E_EA.*B;
        B = np.where(gt == 0, 2 - np.exp(np.log(0.5) / 5 * Dst), np.ones_like(gt))
        Ew = MIN_E_EA * B

        # TPw = sum(dGT(:)) - sum(sum(Ew(GT)));
        # FPw = sum(sum(Ew(~GT)));
        TPw = np.sum(gt) - np.sum(Ew[gt == 1])
        FPw = np.sum(Ew[gt == 0])

        # R = 1- mean2(Ew(GT)); %Weighed Recall
        # P = TPw./(eps+TPw+FPw); %Weighted Precision
        R = 1 - np.mean(Ew[gt])
        P = TPw / (self.eps + TPw + FPw)

        # % Q = (1+Beta^2)*(R*P)./(eps+R+(Beta.*P));
        Q = (1 + self.beta) * R * P / (self.eps + R + self.beta * P)

        return Q

    def show(self):
        return np.mean(self.scores_list)


