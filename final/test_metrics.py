import unittest
import numpy as np
from metrics import evaluate

class MetricTests(unittest.TestCase):
    def target(self,hard=False):
        return dict(boxes=np.array([[0,0,10,10]]), labels=np.array([1]),difficult=np.array([hard]))
    def pred(self, boxes, scores):
        return dict(boxes=np.array(boxes).reshape(-1,4),scores=np.array(scores),labels=np.ones(len(scores)))
    def test_perfect(self):
        r=evaluate([self.pred([[0,0,10,10]],[.9])],[self.target()],['a','b'])
        self.assertEqual(r['mAP50'],1);self.assertIsNone(r['AP50']['b'])
    def test_false_positive_before_true(self):
        r=evaluate([self.pred([[20,20,30,30],[0,0,10,10]],[.9,.8])],[self.target()],['a'])
        self.assertEqual(r['mAP50'],.5)
    def test_empty_prediction(self):
        self.assertEqual(evaluate([self.pred([],[])],[self.target()],['a'])['mAP50'],0)
    def test_difficult_ignored(self):
        r=evaluate([self.pred([[0,0,10,10]],[.99]),self.pred([[0,0,10,10]],[.8])],
                   [self.target(True),self.target()],['a'],True)
        self.assertEqual(r['mAP50'],1)
    def test_duplicate_does_not_match_twice(self):
        r=evaluate([self.pred([[0,0,10,10],[0,0,10,10]],[.9,.8]), self.pred([[0,0,10,10]],[.7])],
                   [self.target(),self.target()],['a'])
        self.assertAlmostEqual(r['mAP50'],5/6)
if __name__=='__main__': unittest.main()
