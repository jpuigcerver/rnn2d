local LSTM, parent = torch.class('rnn2d.LSTM', 'nn.Module')

LSTM.workspace_inference_size = { }
LSTM.workspace_training_size = { }
LSTM.reserve_size = { }
LSTM.fw_inference = { }
LSTM.fw_training = { }
LSTM.bw_data = { }
LSTM.bw_param = { }

if rnn2d.cpu ~= nil then
  LSTM.workspace_inference_size['torch.FloatTensor']      =
    rnn2d.cpu.rnn2d_lstm_cpu_float_inference_workspace_size
  LSTM.workspace_inference_size['torch.DoubleTensor']     =
    rnn2d.cpu.rnn2d_lstm_cpu_double_inference_workspace_size
  LSTM.workspace_training_size['torch.FloatTensor']      =
    rnn2d.cpu.rnn2d_lstm_cpu_float_training_workspace_size
  LSTM.workspace_training_size['torch.DoubleTensor']     =
    rnn2d.cpu.rnn2d_lstm_cpu_double_training_workspace_size
  LSTM.reserve_size['torch.FloatTensor']     =
    rnn2d.cpu.rnn2d_lstm_cpu_float_training_reserve_size
  LSTM.reserve_size['torch.DoubleTensor']    =
    rnn2d.cpu.rnn2d_lstm_cpu_double_training_reserve_size
  LSTM.fw_inference['torch.FloatTensor']     =
    rnn2d.cpu.rnn2d_lstm_cpu_float_fw_inference
  LSTM.fw_inference['torch.DoubleTensor']    =
    rnn2d.cpu.rnn2d_lstm_cpu_double_fw_inference
  LSTM.fw_training['torch.FloatTensor']      =
    rnn2d.cpu.rnn2d_lstm_cpu_float_fw_training
  LSTM.fw_training['torch.DoubleTensor']     =
    rnn2d.cpu.rnn2d_lstm_cpu_double_fw_training
  LSTM.bw_data['torch.FloatTensor']      =
    rnn2d.cpu.rnn2d_lstm_cpu_float_bw_data
  LSTM.bw_data['torch.DoubleTensor']     =
    rnn2d.cpu.rnn2d_lstm_cpu_double_bw_data
  LSTM.bw_param['torch.FloatTensor']      =
    rnn2d.cpu.rnn2d_lstm_cpu_float_bw_param
  LSTM.bw_param['torch.DoubleTensor']     =
    rnn2d.cpu.rnn2d_lstm_cpu_double_bw_param
end

if rnn2d.gpu ~= nil then
  LSTM.workspace_inference_size['torch.CudaTensor']       =
    rnn2d.gpu.rnn2d_lstm_gpu_float_inference_workspace_size
  LSTM.workspace_inference_size['torch.CudaDoubleTensor'] =
    rnn2d.gpu.rnn2d_lstm_gpu_double_inference_workspace_size
  LSTM.workspace_training_size['torch.CudaTensor']       =
    rnn2d.gpu.rnn2d_lstm_gpu_float_training_workspace_size
  LSTM.workspace_training_size['torch.CudaDoubleTensor'] =
    rnn2d.gpu.rnn2d_lstm_gpu_double_training_workspace_size
  LSTM.reserve_size['torch.CudaTensor']       =
    rnn2d.gpu.rnn2d_lstm_gpu_float_training_reserve_size
  LSTM.reserve_size['torch.CudaDoubleTensor'] =
    rnn2d.gpu.rnn2d_lstm_gpu_double_training_reserve_size
  LSTM.fw_inference['torch.CudaTensor']       =
    rnn2d.gpu.rnn2d_lstm_gpu_float_fw_inference
  LSTM.fw_inference['torch.CudaDoubleTensor'] =
    rnn2d.gpu.rnn2d_lstm_gpu_double_fw_inference
  LSTM.fw_training['torch.CudaTensor']        =
    rnn2d.gpu.rnn2d_lstm_gpu_float_fw_training
  LSTM.fw_training['torch.CudaDoubleTensor']  =
    rnn2d.gpu.rnn2d_lstm_gpu_double_fw_training
  LSTM.bw_data['torch.CudaTensor']       =
    rnn2d.gpu.rnn2d_lstm_gpu_float_bw_data
  LSTM.bw_data['torch.CudaDoubleTensor'] =
    rnn2d.gpu.rnn2d_lstm_gpu_double_bw_data
  LSTM.bw_param['torch.CudaTensor']       =
    rnn2d.gpu.rnn2d_lstm_gpu_float_bw_param
  LSTM.bw_param['torch.CudaDoubleTensor'] =
    rnn2d.gpu.rnn2d_lstm_gpu_double_bw_param
end


local sharedBuffer = {}
local function getSharedBufferForStream(device, stream, name, ttype)
  device = device or cutorch.getDevice()
  stream = stream or cutorch.getStream() -- starts from 0
  if not sharedBuffer[device] then sharedBuffer[device] = {} end
  if not sharedBuffer[device][stream] then sharedBuffer[device][stream] = {} end
  local buf = sharedBuffer[device][stream][name]
  if not buf then
    buf = torch.Tensor()
  end
  sharedBuffer[device][stream][name] = buf:type(ttype)
  return sharedBuffer[device][stream][name]
end


function LSTM:__init(inputSize, hiddenSize)
   parent.__init(self)
   assert(inputSize ~= nil)
   assert(hiddenSize ~= nil)

   self.inputSize = inputSize
   self.hiddenSize = hiddenSize
   self.numDirectionParameters =
     (1 + inputSize + hiddenSize + hiddenSize) * 5 * hiddenSize
   self.numTotalParameters =
     4 * self.numDirectionParameters

   self.weight = torch.Tensor():type(self:type())
   self.output = torch.Tensor():type(self:type())
   self.gradInput = torch.Tensor():type(self:type())
   self.gradWeight = torch.Tensor():type(self:type())
   self.reserve = torch.Tensor():type(self:type())

   self:training()
   self:reset()
end

function LSTM:reset(stdv)
   stdv = stdv or 1.0 / math.sqrt(self.hiddenSize)
   self.weight:resize(self.numTotalParameters)
   self.weight:uniform(-stdv, stdv)
   self.gradWeight:resizeAs(self.weight)
   -- Special initialization for the biases:
   -- All init to 0 except biases of the forget gates
   local biases = self:biases()
   for z=1,4 do
     biases[z]:zero()
     biases[z][2]:fill(1)
     biases[z][3]:fill(1)
   end
end

function LSTM:getWorkspacePtr(H, W, N, D)
  local wss = nil
  -- In which device and stream are we running? Use -1, 0 for CPU.
  local device, stream = -1, 0
  if self:type():find('torch.Cuda') ~= nil then
    device = cutorch.getDevice()
    stream = cutorch.getStream()
  end
  if self.train then wss = LSTM.workspace_training_size[self:type()]
  else wss = LSTM.workspace_inference_size[self:type()] end
  assert(wss ~= nil, ('Unknown size for type %q'):format(self:type()))
  workspaceSize = tonumber(wss(H, W, N, D))
  rnn2d.setSharedWorkspaceSize(workspaceSize, true, device, stream)
  return rnn2d.getSharedWorkspace(device, stream)
end

function LSTM:updateOutput(input)
  assert(input:isContiguous())
  assert(input:dim() == 4, 'Input must have 4 dimensions: N x C x H x W')
  local N, K, H, W = input:size(1), input:size(2), input:size(3), input:size(4)
  local D = self.hiddenSize
  assert(self.inputSize == K, 'Incorrect input size!')

  -- In which device and stream are we running? Use -1, 0 for CPU.
  local device, stream = -1, 0
  if self:type():find('torch.Cuda') ~= nil then
    device = cutorch.getDevice()
    stream = cutorch.getStream()
  end
  -- Make a copy of the input with the appropriate layout.
  local inputShared = getSharedBufferForStream(device, stream, 'input', self:type())
  inputShared:resize(H, W, N, K):copy(input:permute(3, 4, 1, 2))
  -- Get shared output buffer.
  local outputShared = getSharedBufferForStream(device, stream, 'output', self:type())
  outputShared:resize(H, W, N, 4 * D):zero()

  -- Get workspace to do the forward pass
  local wsPtr = self:getWorkspacePtr(H, W, N, D)
  if self.train then
    -- Get reserved space needed to do the forward/backward pass
    local rss = LSTM.reserve_size[self:type()]
    assert(rss ~= nil, ('Unknown size for type %q'):format(self:type()))
    rss = math.ceil(tonumber(rss(H, W, N, D)) / self.reserve:elementSize())
    self.reserve:resize(rss):zero()
    -- Do the forward pass for training
    local fw = LSTM.fw_training[self:type()]
    assert(fw ~= nil, ('Layer not implemented for type %q'):format(self:type()))
    fw(H, W, N, K, D, inputShared:data(), nil, self.weight:data(), outputShared:data(),
       wsPtr, self.reserve:data())
  else
    -- Do the forward pass for inference
    local fw = LSTM.fw_inference[self:type()]
    assert(fw ~= nil, ('Layer not implemented for type %q'):format(self:type()))
    fw(H, W, N, K, D, inputShared:data(), nil, self.weight:data(), outputShared:data(),
       wsPtr)
  end
  -- Convert the output buffer to (N x (4 * D) x H x W)
  self.output = outputShared:permute(3, 4, 1, 2):clone()
  return self.output
end

function LSTM:updateGradInput(input, gradOutput)
  assert(self.train)
  assert(input:isContiguous())
  assert(input:dim() == 4, 'Input must have 4 dimensions: N x K x H x W')
  local N, K, H, W = input:size(1), input:size(2), input:size(3), input:size(4)
  local D = self.hiddenSize
  assert(self.inputSize == K, 'Incorrect input size!')
  assert(gradOutput:isContiguous())
  assert(gradOutput:isSameSizeAs(self.output),
	 'output and gradOutput sizes differ')

  -- In which device and stream are we running? Use -1, 0 for CPU.
  local device, stream = -1, 0
  if self:type():find('torch.Cuda') ~= nil then
    device = cutorch.getDevice()
    stream = cutorch.getStream()
  end
  -- Make a copy of the input with the appropriate layout.
  local inputShared = getSharedBufferForStream(device, stream, 'input', self:type())
  inputShared:resize(H, W, N, K):copy(input:permute(3, 4, 1, 2))
  -- Make a copy of the output with the appropriate layout.
  local outputShared = getSharedBufferForStream(device, stream, 'output', self:type())
  outputShared:resize(H, W, N, 4 * D):copy(self.output:permute(3, 4, 1, 2))
  -- Make a copy of the gradOutput with the appropriate layout.
  local gradOutputShared = getSharedBufferForStream(device, stream, 'gradOutput', self:type())
  gradOutputShared:resize(H, W, N, 4 * D):copy(gradOutput:permute(3, 4, 1, 2))
  -- Get shared gradInput buffer.
  local gradInputShared = getSharedBufferForStream(device, stream, 'gradInput', self:type())
  gradInputShared:resize(H, W, N, K):zero()

  -- Get workspace to do the backward pass
  local wsPtr = self:getWorkspacePtr(H, W, N, D)
  -- Do backward pass through the LSTM cells and gates
  local bw = LSTM.bw_data[self:type()]
  assert(bw ~= nil, ('Layer not implemented for type %q'):format(self:type()))
  bw(H, W, N, K, D, inputShared:data(), nil, self.weight:data(),
     outputShared:data(), gradOutputShared:data(), gradInputShared:data(),
     wsPtr, self.reserve:data())

  -- Convert the gradInput buffer to (N x K x H x W)
  self.gradInput = gradInputShared:permute(3, 4, 1, 2):clone()
  return self.gradInput
end

function LSTM:accGradParameters(input, gradOutput, scale)
  scale = scale or 1
  assert(self.train)
  assert(input:isContiguous())
  assert(input:dim() == 4, 'Input must have 4 dimensions: N x K x H x W')
  local N, K, H, W = input:size(1), input:size(2), input:size(3), input:size(4)
  local D = self.hiddenSize
  assert(self.inputSize == K, 'Incorrect input size!')

  -- In which device and stream are we running? Use -1, 0 for CPU.
  local device, stream = -1, 0
  if self:type():find('torch.Cuda') ~= nil then
    device = cutorch.getDevice()
    stream = cutorch.getStream()
  end
  -- Get the shared input buffer.
  local inputShared = getSharedBufferForStream(device, stream, 'input', self:type())
  -- Get the shared output buffer.
  local outputShared = getSharedBufferForStream(device, stream, 'output', self:type())
  -- Note: we do not need to copy again the data, since updateGradInput is
  -- always called before accGradParameters.

  -- Get workspace to do the backward pass
  local wsPtr = self:getWorkspacePtr(H, W, N, D)
  -- Do the backward pass through the LSTM parameters
  local bw = LSTM.bw_param[self:type()]
  assert(bw ~= nil, ('Layer not implemented for type %q'):format(self:type()))
  bw(H, W, N, K, D, inputShared:data(), outputShared:data(), scale,
     self.gradWeight:data(), wsPtr, self.reserve:data())
end

function LSTM:clearState()
  nn.utils.clear(self, 'reserve')
  return parent.clearState(self)
end

function LSTM:biases()
  local biases = {}
  for z=1,4 do
    local offset = (z - 1) * self.numDirectionParameters
    local p = self.weight.new(
      self.weight:storage(),
      self.weight:storageOffset() + offset,
      torch.LongStorage({5, self.hiddenSize}))
    table.insert(biases, p)
  end
  return biases
end

function LSTM:inputWeights()
  local inputWeights = {}
  for z=1,4 do
    local offset = (z - 1) * self.numDirectionParameters +
      5 * self.hiddenSize
    local p = self.weight.new(
      self.weight:storage(),
      self.weight:storageOffset() + offset,
      torch.LongStorage({self.inputSize, 5, self.hiddenSize}))
    table.insert(inputWeights, p)
  end
  return inputWeights
end

function LSTM:recurrentWeights()
  local recurrentWeights = {}
  for z=1,4 do
    local offsetX = (z - 1) * self.numDirectionParameters +
      5 * self.hiddenSize +
      self.inputSize * 5 * self.hiddenSize
    local offsetY = (z - 1) * self.numDirectionParameters +
      5 * self.hiddenSize +
      self.inputSize * 5 * self.hiddenSize +
      self.hiddenSize * 5 * self.hiddenSize
    local pX = self.weight.new(
      self.weight:storage(),
      self.weight:storageOffset() + offsetX,
      torch.LongStorage({self.hiddenSize, 5, self.hiddenSize}))
    local pY = self.weight.new(
      self.weight:storage(),
      self.weight:storageOffset() + offsetY,
      torch.LongStorage({self.hiddenSize, 5, self.hiddenSize}))
    table.insert(recurrentWeights, {x = pX, y = pY})
  end
  return recurrentWeights
end

return LSTM
