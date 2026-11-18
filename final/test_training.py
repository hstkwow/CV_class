"""验证学习率预热和衰减，防止继续训练时设置失效。"""
import unittest
from main import learning_rate_at, parser

class TrainingTests(unittest.TestCase):
    def test_original_step_schedule(self):
        self.assertAlmostEqual(learning_rate_at(9,300,.005,'step',0,10,30),.005)
        self.assertAlmostEqual(learning_rate_at(100,300,.005,'step',0,10,30),.0015)
        self.assertAlmostEqual(learning_rate_at(200,300,.005,'step',0,10,30),.00045)
    def test_warmup_and_cosine(self):
        values=[learning_rate_at(i,100,.001,'cosine',10,10,10) for i in range(100)]
        self.assertTrue(all(a<b for a,b in zip(values[:10],values[1:10])))
        self.assertAlmostEqual(values[9],.001)
        self.assertTrue(all(a>=b for a,b in zip(values[10:],values[11:])))
        self.assertAlmostEqual(values[-1],.00001)
    def test_short_run(self):
        for count in (1,2,5):
            for i in range(count):
                self.assertGreater(learning_rate_at(i,count,.001,'cosine',500,1,count),0)
    def test_new_options(self):
        a=parser().parse_args(['train','--init-from','a.pt','--schedule','cosine',
                              '--warmup-iters','500','--multiscale','480','576'])
        self.assertEqual(a.init_from,'a.pt');self.assertEqual(a.multiscale,[480,576])

if __name__=='__main__': unittest.main()
