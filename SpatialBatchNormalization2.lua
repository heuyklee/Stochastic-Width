--[[
   This file implements Batch Normalization as described in the paper:
   "Batch Normalization: Accelerating Deep Network Training
                         by Reducing Internal Covariate Shift"
                by Sergey Ioffe, Christian Szegedy

   This implementation is useful for inputs coming from convolution layers.
   For non-convolutional layers, see BatchNormalization.lua

   The operation implemented is:
   y =     ( x - mean(x) )
        -------------------- * gamma + beta
        standard-deviation(x)
   where gamma and beta are learnable parameters.

   The learning of gamma and beta is optional.

   Usage:
   with    learnable parameters: nn.SpatialBatchNormalization(N [,eps] [,momentum])
                                 where N = dimensionality of input
   without learnable parameters: nn.SpatialBatchNormalization(N [,eps] [,momentum], false)

   eps is a small value added to the variance to avoid divide-by-zero.
       Defaults to 1e-5

   In training time, this layer keeps a running estimate of it's computed mean and std.
   The running sum is kept with a default momentum of 0.1 (unless over-ridden)
   In test time, this running mean/std is used to normalize.
]]--
local BN2, parent = torch.class('SpatialBatchNormalization2', 'nn.BatchNormalization')
BN2.__version = 2

-- expected dimension of input
BN2.nDim = 4

function BN2:__init(nOutput, eps, momentum, affine)
   parent.__init(self, nOutput, eps, momentum, affine)
   self.beforeWeight = torch.FloatTensor():resizeAs(self.weight):copy(self.weight)
   self.beforeBias = torch.FloatTensor():resizeAs(self.bias):copy(self.bias)
end

function BN2:saveKernels()
   self.beforeWeight:copy(self.weight)
   self.beforeBias:copy(self.bias)
end

function BN2:loadKernels(seltbl)
   -- self.seltbl내 bypass된 kernel의 idx를 이용 
   for _, idx in ipairs(seltbl) do
      self.weight[idx] = self.beforeWeight[idx]
      self.bias[idx] = self.beforeBias[idx]
   end
end
