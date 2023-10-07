class L1_Regularization:
    def __init__(self, lambda_l1):
        self.lambda_l1 = lambda_l1

    def __call__(self, model_parameters):
        l1_loss = 0
        for param in model_parameters:
            l1_loss += param.abs().sum()
        return self.lambda_l1 * l1_loss