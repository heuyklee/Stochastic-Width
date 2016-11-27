local nn = require 'nn'
local SpatialConvolution2, parent =
    torch.class('SpatialConvolution2', 'nn.SpatialConvolution')
local ffi = require 'ffi'
local errcheck = cudnn.errcheck

local autotunerCache = {}
autotunerCache[1] = {} -- forward
autotunerCache[2] = {} -- backwardFilter
autotunerCache[3] = {} -- backwardData

-- giyobe

-- 실험 시 설정 필요 하이퍼 파라미터====
local BYPASS_RATE = 0
local BYPASS_VERSION = 1
-- =====================================

function SpatialConvolution2:bypass()
   self:determineBypass()
   self:makeBypass()
end

function SpatialConvolution2:setNeighborConv(befConv, nexConv, curConvIdx)
   self.befConv = befConv
   self.nexConv = nexConv
   self.curConvIdx = curConvIdx
end

--[[ depre
function SpatialConvolution2:normalizeGradInput()
-- 구현 옵션 중에서 normalize를 gradInput들에 대해서 시키는 방법도 있겠고,
-- 추후에는 weight를 이용해서 normalize 해볼수도 있겠다.
   if self.sel:sum() ~= 0 then -- bypass 존재 시에만 수행
      self.gradInputMean = 0
      self.selGradInputMean = 0
      local rtbl = torch.range(1,self.nOutputPlane):sort():totable()
      local tmptbl = self.sel:sort():totable()

--for k = 1,self.gradInput:size()[1] do
k = 1
      local j = 1
      for i = 1,self.gradInput:size()[2] do
         if rtbl[i] ~= tmptbl[j] then
            self.gradInputMean = self.gradInputMean + self.gradInput[k][i]:mean()
         else
            self.selGradInputMean = self.selGradInputMean + self.gradInput[k][i]:mean()
            j = j + 1
         end
      end 
--end

      self.gradInputMean = self.gradInputMean / (self.gradInput:size()[2] - self.sel:size()[1])-- / self.gradInput:size()[1]
      self.selGradInputMean = self.selGradInputMean / self.sel:size()[1]-- / self.gradInput:size()[1]

      for k = 1,self.gradInput:size()[1] do
         for i, idx in ipairs(self.seltbl) do
            self.gradInput[k][idx]:mul(self.gradInputMean / self.selGradInputMean) 
         end 
      end
   end
end
--]]

--[[ depre
function SpatialConvolution2:gradOutput2gradInput()
   if self.bypassVersion == 1 then
      local sizes = self.gradInput:size()
      for j = 1,sizes[1] do
         for i, idx in ipairs(self.seltbl) do
	    self.gradInput[j][idx]:add(self.gradOutput[j][idx])
         end
      end
   else
      -- Version == 2 미구현
   end
end
--]]

--[[
function SpatialConvolution2:makeBypass()
   -- load/save current kernel before execute bypass mini-batch
   self:loadKernels()
   self:saveKernels()
   
   -- 직접적으로 kernel 값을 0, 1로 수정해주는 부분
   if self.bypassVersion == 1 then
      for i, idx in ipairs(self.seltbl) do
	 self.weight[idx]:zero()
	 self.weight[idx][idx][2][2] = 1 -- 모두 3x3 kernel을 사용하므로 정 가운데 성분만 1로 set
      end
   else 
      self.fidx = torch.randperm(self.nInputPlane) -- 이전 layer의 어떤 feature를 가져올지
      for i, idx in ipairs(self.seltbl) do
	 self.weight[idx]:zero()
	 self.weight[idx][self.fidx[i] ][2][2] = 1 
      end
   end
end
--]]

-- 1121_1800 이후로 다시 사용하도록...
function SpatialConvolution2:makeBKzero()
   for _, idx in ipairs(self.seltbl) do
      self.weight[idx]:zero()
      self.bias[idx] = 0
   end
end

--[[
function SpatialConvolution2:makeBRKzero()
   -- BK 되는 feature map을 사용하지 않고 학습이 진행되도록 커널의 특정 웨이트 0으로
   for _, idx in ipairs(self.seltbl) do
      for i = 1, self.nOutputPlane do
         self.weight[i][idx]:zero()
      end
   end 
end
--]]

function SpatialConvolution2:setBypassRate(bypassRate)
   self.bypassRate = bypassRate
   return self
end

function SpatialConvolution2:determineBypass()
   if self.bypassRate ~= 0 then
      torch.manualSeed(torch.seed(self.gen))
      local bR = self.bypassRate
      local list = torch.randperm(self.nOutputPlane):cuda() 
      local mask = torch.ByteTensor(self.nOutputPlane):bernoulli(bR):cudaByte() -- bR 확률로 각 요소가 1인 mask
      self.sel:maskedSelect(list, mask) -- mask element == 1 인 위치의 idx만을 모아놓은

      -- make sel tensor to table manually
      -- 위에서 randperm을 이용하여 순서를 섞었으므로, 여기서 math.min을 이용하여
      -- 최대 bypass 개수를 nInputPlane으로 제한하였다.
      -- bypassVersion 1과 2를 모두 지원하기 위해서
      if mask:sum() ~= 0 then
         self.seltbl = self.sel:totable()
      end
   end
end

function SpatialConvolution2:loadKernels()
   -- self.seltbl내 bypass된 kernel의 idx를 이용 
   for _, idx in ipairs(self.seltbl) do
      self.weight[idx]:copy(self.beforeWeight[idx])
      self.bias[idx] = self.beforeBias[idx]
   end
end

function SpatialConvolution2:saveKernels()
   self.beforeWeight:copy(self.weight)
   self.beforeBias:copy(self.bias)
end
-- end giyobe

function SpatialConvolution2:__init(nInputPlane, nOutputPlane,
                            kW, kH, dW, dH, padW, padH, groups)
    local delayedReset = self.reset
    self.reset = function() end
    parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH)
    self.reset = delayedReset
    self.padW = padW or 0
    self.padH = padH or 0
    self.groups = groups or 1
    assert(nInputPlane % self.groups == 0,
           'nInputPlane should be divisible by nGroups')
    assert(nOutputPlane % self.groups == 0,
           'nOutputPlane should be divisible by nGroups')
    self.weight = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kH, kW)
    self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kH, kW)
    self:reset()
    -- should nil for serialization, the reset will still work
    self.reset = nil
    
    -- giyobe
    self.bypassVersion = BYPASS_VERSION
    self.bypassRate = BYPASS_RATE 
    self.sel = torch.CudaTensor()
    self.seltbl = {}
    self.beforeWeight = torch.Tensor():resizeAs(self.weight):copy(self.weight)
    self.beforeBias = torch.Tensor():resizeAs(self.bias):copy(self.bias)
    self.gradOutput = torch.CudaTensor()
    self.gen = torch.Generator()
    -- end giyobe
end

-- if you change the configuration of the module manually, call this
function SpatialConvolution2:resetWeightDescriptors()
    assert(cudnn.typemap[torch.typename(self.weight)], 'Only Cuda supported duh!')
    assert(cudnn.typemap[torch.typename(self.bias)] or not self.bias, 'Only Cuda supported duh!')
    -- for compatibility
    self.groups = self.groups or 1
    -- create filterDescriptor for weight
    self.weightDesc = ffi.new('struct cudnnFilterStruct*[1]')
    errcheck('cudnnCreateFilterDescriptor', self.weightDesc)
    local desc = torch.IntTensor({self.nOutputPlane/self.groups,
                              self.nInputPlane/self.groups,
                              self.kH, self.kW})
    errcheck('cudnnSetFilterNdDescriptor', self.weightDesc[0],
             cudnn.typemap[torch.typename(self.weight)], 'CUDNN_TENSOR_NCHW', 4,
             desc:data());
    local function destroyWDesc(d)
        errcheck('cudnnDestroyFilterDescriptor', d[0]);
    end
    ffi.gc(self.weightDesc, destroyWDesc)

    -- create descriptor for bias
    if self.bias then
        self.biasDesc = cudnn.toDescriptor(self.bias:view(1, self.nOutputPlane,1,1))
    end
end

function SpatialConvolution2:fastest(mode)
    if mode == nil then mode = true end
    self.fastest_mode = mode
    self.iSize = self.iSize or torch.LongStorage(4)
    self.iSize:fill(0)
    return self
end

function SpatialConvolution2:setMode(fmode, bdmode, bwmode)
    if fmode ~= nil then
        self.fmode = fmode
    end
    if bdmode ~= nil then
        self.bdmode = bdmode
    end
    if bwmode ~= nil then
        self.bwmode = bwmode
    end
    self.iSize = self.iSize or torch.LongStorage(4)
    self.iSize:fill(0)
    return self
end

function SpatialConvolution2:resetMode()
    self.fmode = nil
    self.bdmode = nil
    self.bwmode = nil
    return self
end

function SpatialConvolution2:noBias()
   self.bias = nil
   self.gradBias = nil
   return self
end

function SpatialConvolution2:createIODescriptors(input)
    local batch = true
    if input:dim() == 3 then
        input = input:view(1, input:size(1), input:size(2), input:size(3))
        batch = false
    end
    assert(input:dim() == 4 and input:isContiguous());
    self.iSize = self.iSize or torch.LongStorage(4):fill(0)
    if not self.iDesc or not self.oDesc or
        input:size(1) ~= self.iSize[1] or input:size(2) ~= self.iSize[2]
    or input:size(3) ~= self.iSize[3] or input:size(4) ~= self.iSize[4] then
        self.iSize = input:size()

        assert(self.nInputPlane == input:size(2), 'input has to contain: '
                   .. self.nInputPlane
                   .. ' feature maps, but received input of size: '
                   .. input:size(1) .. ' x ' .. input:size(2) ..
                   ' x ' .. input:size(3) .. ' x ' .. input:size(4))

        -- create input descriptor
        local input_slice = input:narrow(2,1,self.nInputPlane/self.groups)
        self.iDesc = cudnn.toDescriptor(input_slice)

        -- create conv descriptor
        self.convDesc = ffi.new('struct cudnnConvolutionStruct*[1]')
        errcheck('cudnnCreateConvolutionDescriptor', self.convDesc)
        self.padH, self.padW = self.padH or 0, self.padW or 0
        local pad = torch.IntTensor({self.padH, self.padW})
        local stride = torch.IntTensor({self.dH, self.dW})
        local upscale = torch.IntTensor({1,1})
        errcheck('cudnnSetConvolutionNdDescriptor', self.convDesc[0],
                 2, pad:data(),
                 stride:data(), upscale:data(), 'CUDNN_CROSS_CORRELATION',
                 cudnn.configmap(torch.type(self.weight)));
        local function destroyConvDesc(d)
            errcheck('cudnnDestroyConvolutionDescriptor', d[0]);
        end
        ffi.gc(self.convDesc, destroyConvDesc)

        -- get output shape, resize output
        local oSize = torch.IntTensor(4)
        local oSizeD = oSize:data()
        errcheck('cudnnGetConvolutionNdForwardOutputDim',
                 self.convDesc[0], self.iDesc[0],
                 self.weightDesc[0], 4, oSizeD)
        oSize[2] = oSize[2] * self.groups
        self.output:resize(oSize:long():storage())

        -- create descriptor for output
        local output_slice = self.output:narrow(2,1,self.nOutputPlane/self.groups)
        self.oDesc = cudnn.toDescriptor(output_slice)
        self.oDescForBias = cudnn.toDescriptor(self.output)

        -----------------------------------------------------------------------
        local function shape(x)
            local sz = x:size()
            local str = ''
            for i=1,sz:size() do
                str = str .. sz[i] .. 'x'
            end
            if #str > 0 then
                str = str:sub(1, #str-1)
            end
            return str
        end
        local autotunerHash = shape(self.weight) .. ';'
            .. shape(input_slice) .. ';'
            .. shape(output_slice)

        local maxBufSize = 0

        -- create forwardAlgorithm descriptors
        local algType = ffi.new("cudnnConvolutionFwdAlgo_t[?]", 1)
        local algSearchMode = 'CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT'
        local algWorkspaceLimit = self.workspace_limit
           or (self.nInputPlane * self.kH * self.kW * cudnn.sizeof(self.weight))

        if self.fastest_mode or cudnn.fastest == true then
            algSearchMode = 'CUDNN_CONVOLUTION_FWD_PREFER_FASTEST'
        end

        if cudnn.benchmark then -- the manual auto-tuner is run
            if autotunerCache[1][autotunerHash] then
                algType[0] = autotunerCache[1][autotunerHash]
                if cudnn.verbose then
                   print('Autotuning SC FW: using cached algo = ', algType[0], ' for: ', autotunerHash)
                end
            else
                local perfResults = ffi.new("cudnnConvolutionFwdAlgoPerf_t[?]", 1)
                local intt = torch.IntTensor(1);
                errcheck('cudnnFindConvolutionForwardAlgorithm',
                         cudnn.getHandle(),
                         self.iDesc[0], self.weightDesc[0],
                         self.convDesc[0], self.oDesc[0],
                         1, intt:data(), perfResults)
                algType[0] = perfResults[0].algo
                autotunerCache[1][autotunerHash] = perfResults[0].algo
                if cudnn.verbose then
                    print(string.format(
                              "\nAutotuning SC     Forward: Time: %3.5f Memory: %8d Algorithm: %d"
                                  .. " Weight: %15s Input: %15s Output: %15s",
                              perfResults[0].time, tonumber(perfResults[0].memory),
                              tonumber(perfResults[0].algo),
                              shape(self.weight), shape(input_slice),
                              shape(output_slice)))
                end
            end
        else
            errcheck('cudnnGetConvolutionForwardAlgorithm',
                     cudnn.getHandle(),
                     self.iDesc[0], self.weightDesc[0],
                     self.convDesc[0], self.oDesc[0],
                     algSearchMode, algWorkspaceLimit, algType)
        end
        algType[0] = self.fmode or algType[0]
        self.fwdAlgType = algType
        local bufSize = torch.LongTensor(1)
        errcheck('cudnnGetConvolutionForwardWorkspaceSize',
                 cudnn.getHandle(),
                 self.iDesc[0], self.weightDesc[0],
                 self.convDesc[0], self.oDesc[0],
                 algType[0], bufSize:data())
        maxBufSize = math.max(maxBufSize, bufSize[1])

        -- create backwardFilterAlgorithm descriptors
        local algType = ffi.new("cudnnConvolutionBwdFilterAlgo_t[?]", 1)
        local algSearchMode = 'CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE'
        local algWorkspaceLimit = self.workspace_limit
           or (self.nInputPlane * self.kH * self.kW * cudnn.sizeof(self.weight))
        if self.fastest_mode  or cudnn.fastest == true then
            algSearchMode = 'CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST'
        end

        if cudnn.benchmark then -- the manual auto-tuner is run
            if autotunerCache[2][autotunerHash] then
                algType[0] = autotunerCache[2][autotunerHash]
                if cudnn.verbose then
                   print('Autotuning SC BW: using cached algo = ', algType[0], ' for: ', autotunerHash)
                end
            else
                local perfResults = ffi.new("cudnnConvolutionBwdFilterAlgoPerf_t[?]", 1)
                local intt = torch.IntTensor(1);
                errcheck('cudnnFindConvolutionBackwardFilterAlgorithm',
                         cudnn.getHandle(),
                         self.iDesc[0], self.oDesc[0],
                         self.convDesc[0], self.weightDesc[0],
                         1, intt:data(), perfResults)
                algType[0] = perfResults[0].algo
                autotunerCache[2][autotunerHash] = perfResults[0].algo
                if cudnn.verbose then
                    print(string.format(
                              "Autotuning backwardFilter: Time: %3.5f Memory: %8d Algorithm: %d"
                                  .. " Weight: %15s Input: %15s Output: %15s",
                              perfResults[0].time, tonumber(perfResults[0].memory),
                              tonumber(perfResults[0].algo),
                              shape(self.weight), shape(input_slice),
                              shape(output_slice)))
                end
            end
        else
            errcheck('cudnnGetConvolutionBackwardFilterAlgorithm',
                     cudnn.getHandle(),
                     self.iDesc[0], self.oDesc[0],
                     self.convDesc[0], self.weightDesc[0],
                     algSearchMode, algWorkspaceLimit, algType)
        end
        algType[0] = self.bwmode or algType[0]
        self.bwdFilterAlgType = algType
        local bufSize = torch.LongTensor(1)
        errcheck('cudnnGetConvolutionBackwardFilterWorkspaceSize',
                 cudnn.getHandle(),
                 self.iDesc[0], self.oDesc[0],
                 self.convDesc[0], self.weightDesc[0],
                 algType[0], bufSize:data())
        maxBufSize = math.max(maxBufSize, bufSize[1])

        -- create backwardDataAlgorithm descriptors
        local algType = ffi.new("cudnnConvolutionBwdDataAlgo_t[?]", 1)
        local algSearchMode = 'CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE'
        local algWorkspaceLimit = self.workspace_limit
           or (self.nInputPlane * self.kH * self.kW * cudnn.sizeof(self.weight))
        if self.fastest_mode or cudnn.fastest == true then
            algSearchMode = 'CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST'
        end
        if cudnn.benchmark then -- the manual auto-tuner is run
            if autotunerCache[3][autotunerHash] then
                algType[0] = autotunerCache[3][autotunerHash]
                if cudnn.verbose then
                   print('Autotuning SC BWD: using cached algo = ', algType[0], ' for: ', autotunerHash)
                end
            else
                local perfResults = ffi.new("cudnnConvolutionBwdDataAlgoPerf_t[?]", 1)
                local intt = torch.IntTensor(1);
                errcheck('cudnnFindConvolutionBackwardDataAlgorithm',
                         cudnn.getHandle(),
                         self.weightDesc[0], self.oDesc[0],
                         self.convDesc[0], self.iDesc[0],
                         1, intt:data(), perfResults)
                algType[0] = perfResults[0].algo
                autotunerCache[3][autotunerHash] = perfResults[0].algo
                if cudnn.verbose then
                    print(string.format(
                              "Autotuning   backwardData: Time: %3.5f Memory: %8d Algorithm: %d"
                                  .. " Weight: %15s Input: %15s Output: %15s\n",
                              perfResults[0].time, tonumber(perfResults[0].memory),
                              tonumber(perfResults[0].algo),
                              shape(self.weight), shape(input_slice),
                              shape(output_slice)))
                end
            end
        else
            errcheck('cudnnGetConvolutionBackwardDataAlgorithm',
                     cudnn.getHandle(),
                     self.weightDesc[0], self.oDesc[0],
                     self.convDesc[0], self.iDesc[0],
                     algSearchMode, algWorkspaceLimit, algType)
        end
        algType[0] = self.bdmode or algType[0]
        self.bwdDataAlgType = algType
        local bufSize = torch.LongTensor(1)
        errcheck('cudnnGetConvolutionBackwardDataWorkspaceSize',
                 cudnn.getHandle(),
                 self.weightDesc[0], self.oDesc[0],
                 self.convDesc[0], self.iDesc[0],
                 algType[0], bufSize:data())
        maxBufSize = math.max(maxBufSize, bufSize[1])

        self.extraBuffer = self.extraBuffer or cudnn.getSharedWorkspace()
        self.extraBuffer = self.extraBuffer:cuda() -- always force float
        self.extraBufferSizeInBytes =
           self.extraBuffer:nElement() * 4 -- extraBuffer is always float
        if maxBufSize > self.extraBufferSizeInBytes then
           self.extraBuffer:resize(math.ceil(maxBufSize / 4))
           self.extraBufferSizeInBytes = maxBufSize
        end

        -----------------------------------------------------------------------
        -- create offsets for groups
        local iH, iW = input:size(3), input:size(4)
        local kH, kW = self.kH, self.kW
        local oH, oW = oSize[3], oSize[4]
        self.input_offset = self.nInputPlane / self.groups * iH * iW
        self.output_offset = self.nOutputPlane / self.groups * oH * oW
        self.weight_offset = self.nInputPlane / self.groups * self.nOutputPlane / self.groups * kH * kW

        if not batch then
            self.output = self.output:view(self.output:size(2),
                                           self.output:size(3),
                                           self.output:size(4))
        end
    end
end

local function makeContiguous(self, input, gradOutput)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:typeAs(input):resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput and not gradOutput:isContiguous() then
      self._gradOutput = self._gradOutput or gradOutput.new()
      self._gradOutput:typeAs(gradOutput):resizeAs(gradOutput):copy(gradOutput)
      gradOutput = self._gradOutput
   end
   return input, gradOutput
end

function SpatialConvolution2:updateOutput(input)
   if not self.weightDesc then self:resetWeightDescriptors() end
   input = makeContiguous(self, input)
   self:createIODescriptors(input)

   for g = 0, self.groups - 1 do
       errcheck('cudnnConvolutionForward', cudnn.getHandle(),
                cudnn.scalar(input, 1),
                self.iDesc[0], input:data() + g*self.input_offset,
                self.weightDesc[0], self.weight:data() + g*self.weight_offset,
                self.convDesc[0], self.fwdAlgType[0],
                self.extraBuffer:data(), self.extraBufferSizeInBytes,
                cudnn.scalar(input, 0),
                self.oDesc[0], self.output:data() + g*self.output_offset);
   end

   -- add bias
   if self.bias then
       errcheck('cudnnAddTensor', cudnn.getHandle(),
                cudnn.scalar(input, 1), self.biasDesc[0], self.bias:data(),
                cudnn.scalar(input, 1), self.oDescForBias[0], self.output:data())
   end

   -- giyobe
   if self.befConv and self.bypassRate ~= 0 then
      for _, idx in ipairs(self.seltbl) do
         self.output:narrow(2,idx,1):copy(self.befConv.output:narrow(2,idx,1))
      end
   end   
   -- end giyobe

   return self.output
end

function SpatialConvolution2:updateGradInput(input, gradOutput)
    if not self.gradInput then return end
    self.gradInput:resizeAs(input)
    input, gradOutput = makeContiguous(self, input, gradOutput)

    --[[ giyobe
    if self.nexConv and self.nexConv.bypassRate ~= 0 then
       for _,idx in ipairs(self.nexConv.seltbl) do
          gradOutput:narrow(2,idx,1):add(self.nexConv.gradOutput:narrow(2,idx,1)):mul(0.5)
          -- gradOutput:narrow(2,idx,1):add(self.nexConv.gradOutput:narrow(2,idx,1))
          -- gradOutput:narrow(2,idx,1):copy(self.nexConv.gradOutput:narrow(2,idx,1))
       end 
    end
    --]]
    -- self.gradOutput:resizeAs(gradOutput):copy(gradOutput)
    -- print(self.gradOutput:sum())
    -- self.gradOutput = torch.CudaTensor():resizeAs(gradOutput):copy(gradOutput)
    -- self.gradOutput:resizeAs(gradOutput):copy(gradOutput)
    -- end giyobe

    assert(gradOutput:dim() == 3 or gradOutput:dim() == 4, 'gradOutput has to be 3D or 4D');
    if not self.weightDesc then self:resetWeightDescriptors() end
    self:createIODescriptors(input)

    for g = 0,self.groups - 1 do
        errcheck('cudnnConvolutionBackwardData', cudnn.getHandle(),
                 cudnn.scalar(input, 1),
                 self.weightDesc[0], self.weight:data() + g*self.weight_offset,
                 self.oDesc[0], gradOutput:data() + g*self.output_offset,
                 self.convDesc[0],
                 self.bwdDataAlgType[0],
                 self.extraBuffer:data(), self.extraBufferSizeInBytes,
                 cudnn.scalar(input, 0),
                 self.iDesc[0], self.gradInput:data() + g*self.input_offset);
    end

    return self.gradInput
end

function SpatialConvolution2:accGradParameters(input, gradOutput, scale)
    self.scaleT = self.scaleT or self.weight.new(1)
    -- this line forces this member to always be on CPU (needed for cudnn)
    self.scaleT = torch.type(self.weight) == 'torch.CudaDoubleTensor'
       and self.scaleT:double() or self.scaleT:float()
    scale = scale or 1.0
    self.scaleT[1] = scale

    input, gradOutput = makeContiguous(self, input, gradOutput)
    assert(gradOutput:dim() == 3 or gradOutput:dim() == 4, 'gradOutput has to be 3D or 4D');
    if not self.weightDesc then self:resetWeightDescriptors() end
    self:createIODescriptors(input)

    -- gradBias
    if self.bias then
        errcheck('cudnnConvolutionBackwardBias', cudnn.getHandle(),
                 self.scaleT:data(),
                 self.oDescForBias[0], gradOutput:data(),
                 cudnn.scalar(input, 1),
                 self.biasDesc[0], self.gradBias:data())
    end

    for g = 0, self.groups - 1 do
        -- gradWeight
        errcheck('cudnnConvolutionBackwardFilter', cudnn.getHandle(),
                 self.scaleT:data(),
                 self.iDesc[0], input:data() + g*self.input_offset,
                 self.oDesc[0], gradOutput:data() + g*self.output_offset,
                 self.convDesc[0],
                 self.bwdFilterAlgType[0],
                 self.extraBuffer:data(), self.extraBufferSizeInBytes,
                 cudnn.scalar(input, 1),
                 self.weightDesc[0], self.gradWeight:data() + g*self.weight_offset);
    end
end

function SpatialConvolution2:clearDesc()
    self.weightDesc = nil
    self.biasDesc = nil
    self.convDesc = nil
    self.iDesc = nil
    self.oDesc = nil
    self.oDescForBias = nil
    self.algType = nil
    self.fwdAlgType = nil
    self.bwdDataAlgType = nil
    self.bwdFilterAlgType = nil
    self.extraBuffer = nil
    self.extraBufferSizeInBytes = nil
    self.scaleT = nil
end

function SpatialConvolution2:write(f)
    self:clearDesc()
    local var = {}
    for k,v in pairs(self) do
        var[k] = v
    end
    f:writeObject(var)
end

function SpatialConvolution2:clearState()
   self:clearDesc()
   nn.utils.clear(self, '_input', '_gradOutput')
   return nn.Module.clearState(self)
end
