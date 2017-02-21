local LSTM, parent = torch.class('rnn2d.LSTM', 'nn.Module')

LSTM.workspace_inference_size = {
  ['torch.FloatTensor']      =
    rnn2d.cpu.rnn2d_lstm_cpu_float_inference_workspace_size,
  ['torch.DoubleTensor']     =
    rnn2d.cpu.rnn2d_lstm_cpu_double_inference_workspace_size,
  ['torch.CudaTensor']       =
    rnn2d.gpu.rnn2d_lstm_gpu_float_inference_workspace_size,
  ['torch.CudaDoubleTensor'] =
    rnn2d.gpu.rnn2d_lstm_gpu_double_inference_workspace_size
}

LSTM.workspace_training_size = {
  ['torch.FloatTensor']      =
    rnn2d.cpu.rnn2d_lstm_cpu_float_training_workspace_size,
  ['torch.DoubleTensor']     =
    rnn2d.cpu.rnn2d_lstm_cpu_double_training_workspace_size,
  ['torch.CudaTensor']       =
    rnn2d.gpu.rnn2d_lstm_gpu_float_training_workspace_size,
  ['torch.CudaDoubleTensor'] =
    rnn2d.gpu.rnn2d_lstm_gpu_double_training_workspace_size
}

LSTM.reserve_size = {
  ['torch.FloatTensor']      =
    rnn2d.cpu.rnn2d_lstm_cpu_float_training_reserve_size,
  ['torch.DoubleTensor']     =
    rnn2d.cpu.rnn2d_lstm_cpu_double_training_reserve_size,
  ['torch.CudaTensor']       =
    rnn2d.gpu.rnn2d_lstm_gpu_float_training_reserve_size,
  ['torch.CudaDoubleTensor'] =
    rnn2d.gpu.rnn2d_lstm_gpu_double_training_reserve_size
}

LSTM.fw_inference = {
  ['torch.FloatTensor']      = rnn2d.cpu.rnn2d_lstm_cpu_float_fw_inference,
  ['torch.DoubleTensor']     = rnn2d.cpu.rnn2d_lstm_cpu_double_fw_inference,
  ['torch.CudaTensor']       = rnn2d.gpu.rnn2d_lstm_gpu_float_fw_inference,
  ['torch.CudaDoubleTensor'] = rnn2d.gpu.rnn2d_lstm_gpu_double_fw_inference
}

LSTM.fw_training = {
  ['torch.FloatTensor']      = rnn2d.cpu.rnn2d_lstm_cpu_float_fw_training,
  ['torch.DoubleTensor']     = rnn2d.cpu.rnn2d_lstm_cpu_double_fw_training,
  ['torch.CudaTensor']       = rnn2d.gpu.rnn2d_lstm_gpu_float_fw_training,
  ['torch.CudaDoubleTensor'] = rnn2d.gpu.rnn2d_lstm_gpu_double_fw_training
}

LSTM.bw_workspace = {
  ['torch.FloatTensor']      = rnn2d.cpu.rnn2d_lstm_cpu_float_bw_workspace,
  ['torch.DoubleTensor']     = rnn2d.cpu.rnn2d_lstm_cpu_double_bw_workspace,
  ['torch.CudaTensor']       = rnn2d.gpu.rnn2d_lstm_gpu_float_bw_workspace,
  ['torch.CudaDoubleTensor'] = rnn2d.gpu.rnn2d_lstm_gpu_double_bw_workspace
}

LSTM.bw_input = {
  ['torch.FloatTensor']      = rnn2d.cpu.rnn2d_lstm_cpu_float_bw_input,
  ['torch.DoubleTensor']     = rnn2d.cpu.rnn2d_lstm_cpu_double_bw_input,
  ['torch.CudaTensor']       = rnn2d.gpu.rnn2d_lstm_gpu_float_bw_input,
  ['torch.CudaDoubleTensor'] = rnn2d.gpu.rnn2d_lstm_gpu_double_bw_input
}

LSTM.bw_param = {
  ['torch.FloatTensor']      = rnn2d.cpu.rnn2d_lstm_cpu_float_bw_param,
  ['torch.DoubleTensor']     = rnn2d.cpu.rnn2d_lstm_cpu_double_bw_param,
  ['torch.CudaTensor']       = rnn2d.gpu.rnn2d_lstm_gpu_float_bw_param,
  ['torch.CudaDoubleTensor'] = rnn2d.gpu.rnn2d_lstm_gpu_double_bw_param
}

function LSTM:__init(inputSize, hiddenSize)
   parent.__init(self)
   assert(inputSize ~= nil)
   assert(hiddenSize ~= nil)

   self.inputSize = inputSize
   self.hiddenSize = hiddenSize

   self.weight = torch.Tensor():type(self:type())
   self.output = torch.Tensor():type(self:type())
   self.gradInput = torch.Tensor():type(self:type())
   self.gradWeight = torch.Tensor():type(self:type())
   self.reserve = torch.Tensor():type(self:type())

   self._backprop_workspace_done = false

   self:training()
   self:reset()
end

function LSTM:reset(stdv)
   stdv = stdv or 1.0 / math.sqrt(self.hiddenSize)
   local weightSize =
     4 * (1 + self.inputSize + 2 * self.hiddenSize) * 5 * self.hiddenSize
   self.weight:resize(weightSize)
   self.weight:uniform(-stdv, stdv)
   self.gradWeight:resizeAs(self.weight)
end

function LSTM:makeContiguous(input, gradOutput)
  if not input:isContiguous() or input:type() ~= self:type() then
    self._input = self._input or input.new():type(self:type())
    self._input = self._input:resizeAs(input):copy(input)
    input = self._input
  end
  if gradOutput and (not gradOutput:isContiguous() or
		     gradOutput:type() ~= self:type()) then
    self._gradOutput = self._gradOutput or gradOutput.new():type(self:type())
    self._gradOutput = self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
    gradOutput = self._gradOutput
  end
  return input, gradOutput
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
  rnn2d.setSharedWorkspaceSize(tonumber(wss(H, W, N, D)), true, device, stream)
  return rnn2d.getSharedWorkspace(device, stream)
end

function LSTM:updateOutput(input)
  assert(input:dim() == 4, 'Input must have 4 dimensions: H x W x N x D')
  local H, W, N, K = input:size(1), input:size(2), input:size(3), input:size(4)
  local D = self.hiddenSize
  assert(self.inputSize == K, 'Incorrect input size!')
  local x = self:makeContiguous(input)
  -- TODO(jpuigcerver): Use specialized functions for inference which require
  -- less computation and/or space
  self.output = self.output:resize(H, W, N, 4 * D)
  -- Get workspace to do the forward pass
  local wsPtr = self:getWorkspacePtr(H, W, N, D)
  if self.train then
    -- Get reserved space needed to do the forward/backward pass
    local rss = LSTM.reserve_size[self:type()]
    assert(rss ~= nil, ('Unknown size for type %q'):format(self:type()))
    rss = math.ceil(tonumber(rss(H, W, N, D)) / self.reserve:elementSize())
    self.reserve:resize(rss)
    -- Do the forward pass for training
    local fw = LSTM.fw_training[self:type()]
    assert(fw ~= nil, ('Layer not implemented for type %q'):format(self:type()))
    self.gradInput = self.gradInput:resizeAs(x):zero()
    fw(H, W, N, K, D, x:data(), nil, self.weight:data(), self.output:data(),
       wsPtr, self.reserve:data())
  else
    -- Do the forward pass for inference
    local fw = LSTM.fw_inference[self:type()]
    assert(fw ~= nil, ('Layer not implemented for type %q'):format(self:type()))
    fw(H, W, N, K, D, x:data(), nil, self.weight:data(), self.output:data(),
       wsPtr)
  end
  self._backprop_workspace_done = false
  return self.output
end

function LSTM:backpropCellsAndGates(input, gradOutput)
  assert(self.train)
  assert(input:dim() == 4, 'Input must have 4 dimensions: H x W x N x D')
  local H, W, N, K = input:size(1), input:size(2), input:size(3), input:size(4)
  local D = self.hiddenSize
  assert(self.inputSize == K, 'Incorrect input size!')
  assert(gradOutput:isSameSizeAs(self.output),
	 'output and gradOutput sizes differ')
  local x, dy = self:makeContiguous(input, gradOutput)
  if not self._backprop_workspace_done then
    -- Get workspace to do the backward pass
    local wsPtr = self:getWorkspacePtr(H, W, N, D)
    -- Do backward pass through the LSTM cells and gates
    local bw = LSTM.bw_workspace[self:type()]
    assert(bw ~= nil, ('Layer not implemented for type %q'):format(self:type()))
    bw(H, W, N, K, D, x:data(), nil, self.weight:data(), self.output:data(),
       dy:data(), wsPtr, self.reserve:data())
    self._backprop_workspace_done = true
  end
  return x
end

function LSTM:updateGradInput(input, gradOutput)
  local x = self:backpropCellsAndGates(input, gradOutput)
  local H, W, N, K = x:size(1), x:size(2), x:size(3), x:size(4)
  local D = self.hiddenSize
  -- Get workspace to do the backward pass
  local wsPtr = self:getWorkspacePtr(H, W, N, D)
  -- Do the backward pass through the LSTM input
  local bw = LSTM.bw_input[self:type()]
  assert(bw ~= nil, ('Layer not implemented for type %q'):format(self:type()))
  bw(H, W, N, K, D, self.weight:data(), 1.0, self.gradInput:data(),
     wsPtr, self.reserve:data())
  return self.gradInput
end

function LSTM:accGradParameters(input, gradOutput, scale)
  scale = scale or 1
  local x = self:backpropCellsAndGates(input, gradOutput)
  local H, W, N, K = x:size(1), x:size(2), x:size(3), x:size(4)
  local D = self.hiddenSize
  -- Get workspace to do the backward pass
  local wsPtr = self:getWorkspacePtr(H, W, N, D)
  -- Do the backward pass through the LSTM parameters
  local bw = LSTM.bw_param[self:type()]
  assert(bw ~= nil, ('Layer not implemented for type %q'):format(self:type()))
  bw(H, W, N, K, D, x:data(), self.output:data(), scale, self.gradWeight:data(),
     wsPtr, self.reserve:data())
end

function LSTM:clearState()
  nn.utils.clear(self, '_input', '_gradOutput', 'reserve')
  return parent.clearState(self)
end

return LSTM
