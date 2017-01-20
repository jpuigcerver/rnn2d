local Tile, parent = torch.class('rnn2d.Tile', 'nn.Module')

Tile.fw = {
  ['torch.FloatTensor']      = rnn2d.cpu.rnn2d_tile_cpu_float_fw,
  ['torch.DoubleTensor']     = rnn2d.cpu.rnn2d_tile_cpu_double_fw,
  ['torch.CudaTensor']       = rnn2d.gpu.rnn2d_tile_gpu_float_fw,
  ['torch.CudaDoubleTensor'] = rnn2d.gpu.rnn2d_tile_gpu_double_fw
}

Tile.bw = {
  ['torch.FloatTensor']      = rnn2d.cpu.rnn2d_tile_cpu_float_bw,
  ['torch.DoubleTensor']     = rnn2d.cpu.rnn2d_tile_cpu_double_bw,
  ['torch.CudaTensor']       = rnn2d.gpu.rnn2d_tile_gpu_float_bw,
  ['torch.CudaDoubleTensor'] = rnn2d.gpu.rnn2d_tile_gpu_double_bw
}

function Tile:__init(inputSize, kH, kW)
  parent.__init(self)
  assert(inputSize ~= nil)
  self.inputSize = inputSize
  self.kH = kH or 1
  self.kW = kW or 1
  self.output = torch.Tensor()
  self.gradInput = torch.Tensor()
end

function Tile:makeContiguous(input, gradOutput)
  if not input:isContiguous() then
    self._input = self._input or input.new()
    input = self._input:typeAs(input):resizeAs(input):copy(input)
  end
  if gradOutput and not gradOutput:isContiguous() then
    self._gradOutput = self._gradOutput or gradOutput.new()
    gradOutput = self._gradOutput:typeAs(gradOutput):resizeAs(gradOutput)
      :copy(gradOutput)
  end
  if gradOutput and gradOutput:type() ~= input:type() then
    gradOutput = gradOutput:typeAs(input)
  end
  return input, gradOutput
end

function Tile:updateOutput(input)
  assert(input:dim() == 4, 'Input must have 4 dimensions: H x W x N x D')
  local H, W, N, D = input:size(1), input:size(2), input:size(3), input:size(4)
  assert(D == self.inputSize)
  local fw = Tile.fw[self:type()]
  assert(fw ~= nil, ('Layer not implemented for type %q'):format(self:type()))
  -- Resize output tensor to the appropiate size
  local oH, oW = torch.ceil(H / self.kH), torch.ceil(W / self.kW)
  local oD = self.kH * self.kW * D
  self.output = self.output:typeAs(self:type()):resize(oH, oW, N, oD)
  -- Forward step
  local x = self:makeContiguous(input)
  fw(H, W, N, D, self.kH, self.kW, nil, torch.data(x), torch.data(self.output))
  return self.output
end

function Tile:updateGradInput(input, gradOutput)
  assert(input:dim() == 4, 'Input must have 4 dimensions: H x W x N x D')
  local H, W, N, D = input:size(1), input:size(2), input:size(3), input:size(4)
  assert(D == self.inputSize)
  assert(gradOutput:dim() == 4)
  local bw = Tile.bw[input:type()]
  assert(bw ~= nil)
  local x, dy = self:makeContiguous(input, gradOutput)
  self.gradInput:typeAs(x):resizeAs(x):zero()
  bw(H, W, N, D, self.kH, self.kW, nil, torch.data(dy),
     torch.data(self.gradInput))
  return self.gradInput
end

function Tile:clearState()
  nn.utils.clear(self, '_input', '_gradOutput')
  return parent.clearState(self)
end

return Tile
