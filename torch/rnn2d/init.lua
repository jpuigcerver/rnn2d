require 'torch'
require 'nn'
pcall(require, 'cutorch')

rnn2d = {}
require('rnn2d.ffi')

-- This code was shamesly stolen from cudnn.torch and is useful to reuse the
-- workspace for several LSTM layers.

rnn2d.initialWorkspaceBytes = 1024
local sharedBuffer = {}
local nextBufferSize = {}

-- may reassign currentSize
local function allocateStorage(buf, ifGreater, cudaAlloc)
  if buf.nextSize < 0 then
    buf.nextSize = buf.currentSize
  end

  -- get number of elements in the buf, rounded up
  local elSize = 8
  local newelem = math.ceil(buf.nextSize / elSize)

  if buf.storage then
    if (newelem ~= buf.storage:size()) and
    ((not ifGreater) or newelem > buf.storage:size()) then
      -- resize to just to make sure we return memory
      buf.storage:resize(0)
      buf.storage:resize(newelem)
    end
  else
    -- this is to be replaced with new cutorch tempbuf stuff
    -- may reassign currentSize again
    if cudaAlloc then buf.storage = torch.CudaDoubleStorage(newelem)
    else buf.storage = torch.DoubleStorage(newelem) end
  end

  buf.currentSize = buf.storage:size()*elSize
  buf.data = buf.storage:data()
  buf.nextSize = -1
end

local function sharedBufferForStream(device, stream)
  device = device or cutorch.getDevice()
  stream = stream or cutorch.getStream() -- starts from 0
  if not sharedBuffer[device] then sharedBuffer[device] = {} end
  local buf = sharedBuffer[device][stream]
  if not buf then
    buf = {
      currentSize = rnn2d.initialWorkspaceBytes,
      nextSize = -1
    }
    allocateStorage(buf, false, (device > 0))
    sharedBuffer[device][stream] = buf
  end
  return buf
end

function rnn2d.getSharedWorkspace(device, stream)
  device = device or cutorch.getDevice()
  stream = stream or cutorch.getStream()
  local buf = sharedBufferForStream(device, stream)
  return buf.data, buf.currentSize
end

function rnn2d.adjustSharedWorkspaceSize(bytesDelta, device, stream)
   local buf = sharedBufferForStream(device, stream)
   buf.nextSize = buf.currentSize + bytesDelta
   allocateStorage(buf, false, device > 0)
end

function rnn2d.setNextWorkspaceSize(bytes, device, stream)
   local buf = sharedBufferForStream(device, stream)
   buf.nextSize = bytes
   return buf
end

function rnn2d.setSharedWorkspaceSize(bytes, ifGreater, device, stream)
   bytes = bytes or rnn2d.initialWorkspaceBytes
   local buf = rnn2d.setNextWorkspaceSize(bytes, device, stream)
   allocateStorage(buf, ifGreater, device > 0)
end

-- Load layers
require('rnn2d.Collapse')
require('rnn2d.LSTM')
require('rnn2d.Tile')

return rnn2d
