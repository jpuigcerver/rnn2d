require 'librnn2d_torch'
require 'torch'
@REQUIRE_CUTORCH@
require 'nn'

rnn2d = {}

require('rnn2d.LSTM')

return rnn2d
