
import numpy as np
import cv2



class FeatEx:

    def __init__(self):
        self.chain = []

    def append(self, step):
        self.chain.append(step)

    def process(self, ctx ):
        res = []
        for s in self.chain:
            res += s.process(ctx )
        return res



class Moments():

    def __init__(self, using=None):
        self.using = using

    def process(self, ctx, itemname='result'):
        img = ctx[itemname]
        M = cv2.moments(img, binaryImage=True)
        D = []
        for k, v in M.iteritems():
            if not self.using or k in self.using:
                D.append(v)
        return D  #np.asarray(D)


class HuMoments():

    def __init__(self, using=None):
        self.using = using

    def process(self, ctx, itemname='result'):
        img = ctx[itemname]
        M = cv2.moments(img, binaryImage=True)
        HM = cv2.HuMoments(M)
        D = HM.tolist()
        return D[0]

class Pixels():

    def __init__(self):
        pass

    def process(self, ctx, itemname='result'):
        img = ctx[itemname]
        tmp = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        M = np.reshape(tmp, (1, img.shape[0] * img.shape[1]))
        D = M.tolist()
        return D[0]  #np.asarray(D)

class Corners():

    def __init__(self):
        pass

    def process(self, ctx, itemname='result'):
        img = ctx[itemname]
        fimg = np.float32(img)
        dst = cv2.cornerHarris(fimg, 2, 5, 0.1)
        #cv2.imshow("corners", dst)
        #cv2.waitKey(0)
        M = np.reshape(dst, (1, img.shape[0] * img.shape[1]))
        D = M.tolist()
        return D[0]  #np.asarray(D)

