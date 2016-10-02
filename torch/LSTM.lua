local LSTM, parent = torch.class('rnn2d.LSTM', 'nn.Module')

LSTM.fw = {
  ['torch.FloatTensor'] = THFloatTensor_rnn2d_lstm_fw,
  ['torch.DoubleTensor'] = THDoubleTensor_rnn2d_lstm_fw,
  ['torch.CudaTensor'] = THCudaTensor_rnn2d_lstm_fw,
  ['torch.CudaDoubleTensor'] = THCudaDoubleTensor_rnn2d_lstm_fw
}

LSTM.bw_workspace = {
  ['torch.FloatTensor'] = THFloatTensor_rnn2d_lstm_bw_workspace,
  ['torch.DoubleTensor'] = THDoubleTensor_rnn2d_lstm_bw_workspace,
  ['torch.CudaTensor'] = THCudaTensor_rnn2d_lstm_bw_workspace,
  ['torch.CudaDoubleTensor'] = THCudaDoubleTensor_rnn2d_lstm_bw_workspace
}

LSTM.bw_input = {
  ['torch.FloatTensor'] = THFloatTensor_rnn2d_lstm_bw_input,
  ['torch.DoubleTensor'] = THDoubleTensor_rnn2d_lstm_bw_input,
  ['torch.CudaTensor'] = THCudaTensor_rnn2d_lstm_bw_input,
  ['torch.CudaDoubleTensor'] = THCudaDoubleTensor_rnn2d_lstm_bw_input
}

LSTM.bw_params = {
  ['torch.FloatTensor'] = THFloatTensor_rnn2d_lstm_bw_params,
  ['torch.DoubleTensor'] = THDoubleTensor_rnn2d_lstm_bw_params,
  ['torch.CudaTensor'] = THCudaTensor_rnn2d_lstm_bw_params,
  ['torch.CudaDoubleTensor'] = THCudaDoubleTensor_rnn2d_lstm_bw_params
}

function LSTM:__init(inputSize, hiddenSize)
   parent.__init(self)

   self.inputSize = inputSize
   self.hiddenSize = hiddenSize

   self.weight = torch.Tensor()
   self.workspace = torch.Tensor()
   self.output = torch.Tensor()
   self.gradInput = torch.Tensor()
   self.gradWeight = torch.Tensor()
   self.gradWorkspace = torch.Tensor()

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

function LSTM:prepareDims(input)
  -- This assumes that all the inputs have the same size
  if input:type() == 'torch.CudaTensor' or
  input:type() == 'torch.CudaDoubleTensor' then
    return torch.CudaIntTensor()
  else
    return torch.IntTensor()
  end
end

function LSTM:updateOutput(input)
  assert(input:dim() == 4,
	 'Input must have 4 dimensions: height, width, miniBatch, inputSize')
  assert(input:size(4) == self.inputSize, 'Incorrect input size!')
  local x = self:makeContiguous(input)
  local dims = self:prepareDims(x)
  self._backprop_workspace_done = false
  self.output:resize(x:size(1), x:size(2), x:size(3), 4 * self.hiddenSize)
  self.workspace:resize(self.output:nElement() * 6)
  LSTM.fw[self._type](x, dims, self.weight, self.output, self.workspace)
  return self.output
end

function LSTM:backpropWorkspace(input, gradOutput)
  assert(self.train)
  assert(input:dim() == 4,
	 'Input must have 4 dimensions: height, width, miniBatch, inputSize')
  assert(input:size(4) == self.inputSize, 'Incorrect input size!')
  assert(gradOutput:isSameSizeAs(self.output),
	 'output and gradOutput sizes differ')
  local x, dy = self:makeContiguous(input, gradOutput)
  if not self._backprop_workspace_done then
    local dims = self:prepareDims(x)
    self.gradWorkspace:resizeAs(self.workspace):zero()
    LSTM.bw_workspace[self._type](
      x, dims, self.weight, self.output, self.workspace, dy, self.gradWorkspace)
    self._backprop_workspace_done = true
  end
  return x, dy
end

function LSTM:updateGradInput(input, gradOutput)
  x, dy = self:backpropWorkspace(input, gradOutput)
  self.gradInput:resizeAs(x):zero()
  LSTM.bw_input[self._type](self.weight, self.gradWorkspace, self.gradInput, 1)
  return self.gradInput
end

function LSTM:accGradParameters(input, gradOutput, scale)
  x, dy = self:backpropWorkspace(input, gradOutput)
  scale = scale or 1
  LSTM.bw_params[self._type](x, self.output, self.gradWorkspace,
			     self.gradWeight, scale)
end

function LSTM:clearState()
  nn.utils.clear(self, '_input', '_gradOutput', 'workspace', 'gradWorkspace')
  return parent.clearState(self)
end
