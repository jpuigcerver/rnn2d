local Collapse, parent = torch.class('rnn2d.Collapse', 'nn.Module')

function Collapse:__init(op, dimension, narrow)
  parent.__init(self)
  self.op = op
  self.dimension = dimension
  self.narrow = narrow or nil
  assert(self.op == 'sum' or self.op == 'mean',
	 ('Collapse operation %q is not implemented!'):format(op))
  assert(self.dimension ~= nil)
  self.output = torch.Tensor():type(self:type())
  self.gradInput = torch.Tensor():type(self:type())
end

function Collapse:updateOutput(input)
  local dim = self.dimension < 0 and input:dim() + self.dimension + 1 or self.dimension
  local len = self.narrow or input:size(dim)
  local n = input:size(dim) / len
  assert(math.floor(n) == n, ('Input dimension %d is not divisible by %d!'):format(dim, len))
  -- Initialize output to zeros
  self.output = self.output:resizeAs(input:narrow(dim, 1, len)):zero()
  -- Compute output
  for i=1,n do
    if self.op == 'sum' then
      self.output:add(input:narrow(dim, (i - 1) * len + 1, len))
    elseif self.op == 'mean' then
      self.output:add(1.0 / n, input:narrow(dim, (i - 1) * len + 1, len))
    else
      error(('Collapse operation %q not implemented!'):format(self.op))
    end
  end
  return self.output
end

function Collapse:updateGradInput(input, gradOutput)
  local dim = self.dimension < 0 and input:dim() + self.dimension + 1 or self.dimension
  local len = self.narrow or input:size(dim)
  local n = input:size(dim) / len
  assert(math.floor(n) == n, ('Input dimension %d is not divisible by %d!'):format(dim, len))
  -- Initialize gradInput to zeros
  self.gradInput = self.gradInput:resizeAs(input):zero()
  -- Compute gradInput
  for i=1,n do
    if self.op == 'sum' then
      self.gradInput:narrow(dim, (i - 1) * len + 1, len):copy(gradOutput)
    elseif self.op == 'mean' then
      self.gradInput:narrow(dim, (i - 1) * len + 1, len):copy(gradOutput):div(n)
    else
      error(('Collapse operation %q not implemented!'):format(self.op))
    end
  end
  return self.gradInput
end

return Collapse
