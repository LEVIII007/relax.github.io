# %% [code] {"execution":{"iopub.status.busy":"2024-01-26T20:44:50.434727Z","iopub.execute_input":"2024-01-26T20:44:50.435190Z","iopub.status.idle":"2024-01-26T20:44:50.762471Z","shell.execute_reply.started":"2024-01-26T20:44:50.435158Z","shell.execute_reply":"2024-01-26T20:44:50.761384Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2024-01-26T20:44:50.764490Z","iopub.execute_input":"2024-01-26T20:44:50.764987Z","iopub.status.idle":"2024-01-26T20:45:02.374046Z","shell.execute_reply.started":"2024-01-26T20:44:50.764954Z","shell.execute_reply":"2024-01-26T20:45:02.372894Z"}}
!pip install chess

# %% [code] {"execution":{"iopub.status.busy":"2024-01-26T20:45:02.375499Z","iopub.execute_input":"2024-01-26T20:45:02.375819Z","iopub.status.idle":"2024-01-26T20:45:02.382170Z","shell.execute_reply.started":"2024-01-26T20:45:02.375789Z","shell.execute_reply":"2024-01-26T20:45:02.381146Z"}}
import gc
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
import chess

# %% [code] {"execution":{"iopub.status.busy":"2024-01-26T20:45:02.384785Z","iopub.execute_input":"2024-01-26T20:45:02.385102Z","iopub.status.idle":"2024-01-26T20:45:02.397088Z","shell.execute_reply.started":"2024-01-26T20:45:02.385074Z","shell.execute_reply":"2024-01-26T20:45:02.395433Z"}}
board = chess.Board()  #i can give fen string as input to make chess bot
print(board)

# %% [code] {"execution":{"iopub.status.busy":"2024-01-26T20:45:02.398357Z","iopub.execute_input":"2024-01-26T20:45:02.398675Z","iopub.status.idle":"2024-01-26T20:45:02.585308Z","shell.execute_reply.started":"2024-01-26T20:45:02.398646Z","shell.execute_reply":"2024-01-26T20:45:02.583414Z"}}
with open('path_to_your_model.pl', 'rb') as file:
    model = pickle.load(file)

# %% [code] {"execution":{"iopub.status.busy":"2024-01-26T20:45:02.586251Z","iopub.status.idle":"2024-01-26T20:45:02.586705Z","shell.execute_reply.started":"2024-01-26T20:45:02.586489Z","shell.execute_reply":"2024-01-26T20:45:02.586509Z"}}
# Create a mapping from standard notation to custom notation
def return_mat(board):
    notation_mapping = {   #standard to sidhu notation
    'r': 'b-rook',
    'n': 'b-knight',
    'b': 'b-bishop',
    'q': 'b-queen',
    'k': 'b-king',
    'p': 'b-pawn',
    'R': 'w-rook',
    'N': 'w-knight',
    'B': 'w-bishop',
    'Q': 'w-queen',
    'K': 'w-king',
    'P': 'w-pawn',
    '.': None
}
board_str = str(board)
# Replace the standard notation with custom notation
new_board_str = ''
for char in board_str:
    if char in notation_mapping:
        new_board_str += str(notation_mapping[char]) + ' '
    else:
        new_board_str += char

# Print the modified board string
print(new_board_str)

# %% [code] {"execution":{"iopub.status.busy":"2024-01-26T20:45:02.588667Z","iopub.status.idle":"2024-01-26T20:45:02.589291Z","shell.execute_reply.started":"2024-01-26T20:45:02.589078Z","shell.execute_reply":"2024-01-26T20:45:02.589107Z"}}
def input_board(board):
    reverse_notation_mapping = {
    'b-rook': 'r',
    'b-knight': 'n',
    'b-bishop': 'b',
    'b-queen': 'q',
    'b-king': 'k',
    'b-pawn': 'p',
    'w-rook': 'R',
    'w-knight': 'N',
    'w-bishop': 'B',
    'w-queen': 'Q',
    'w-king': 'K',
    'w-pawn': 'P',
    None: '.'
}
    for i in range(8):
        for j in range(8):
            board[i][j] = reverse_notation_mapping[board[i][j]]
    fen = ''
    for row in reversed(board):  # 8th row to 1st row
        empty_count = 0
        for square in row:  # 'a' to 'h'
            if square == '.':  # Empty square
                empty_count += 1
            else:  # Square is occupied by a piece
                if empty_count > 0:
                    fen += str(empty_count)
                    empty_count = 0
                fen += square
        if empty_count > 0:
            fen += str(empty_count)
        fen += '/'
    return fen[:-1]  # Remove the trailing '/'
    return board

# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2024-01-26T20:45:02.590478Z","iopub.status.idle":"2024-01-26T20:45:02.591080Z","shell.execute_reply.started":"2024-01-26T20:45:02.590871Z","shell.execute_reply":"2024-01-26T20:45:02.590898Z"}}
def checkmate_single(board):
    board = board.copy()
    legal_moves = list(board.legal_moves)
    for move in legal_moves:
        board.push_uci(str(move))
        if board.is_checkmate():
            move = board.pop()
            return move
        _ = board.pop()
    return None

# %% [code] {"execution":{"iopub.status.busy":"2024-01-26T20:45:02.592224Z","iopub.status.idle":"2024-01-26T20:45:02.592727Z","shell.execute_reply.started":"2024-01-26T20:45:02.592488Z","shell.execute_reply":"2024-01-26T20:45:02.592511Z"}}
def distribution_over_moves(vals):
    probs = np.array(vals)
    probs = np.exp(probs)
    probs = probs / probs.sum()
    probs = probs ** 3
    probs = probs / probs.sum()
    return probs

# %% [code] {"execution":{"iopub.status.busy":"2024-01-26T20:45:02.594041Z","iopub.status.idle":"2024-01-26T20:45:02.594945Z","shell.execute_reply.started":"2024-01-26T20:45:02.594715Z","shell.execute_reply":"2024-01-26T20:45:02.594739Z"}}
def predict(x):
    model.eval()
    with torch.no_grad():
        outputs = model(x)
        return outputs.cpu().numpy()

# %% [code] {"execution":{"iopub.status.busy":"2024-01-26T20:45:02.596258Z","iopub.status.idle":"2024-01-26T20:45:02.597444Z","shell.execute_reply.started":"2024-01-26T20:45:02.597141Z","shell.execute_reply":"2024-01-26T20:45:02.597170Z"}}
def choose_move(board,color):
    board = input_board(board)

    legal_moves = list(board.legal_moves)

    move = checkmate_single(board)   #checking if single move me possible hai check mate

    if move is not None:   #if yes..kardo
        return move
    
    x = torch.Tensor(board_2_rep(board)).float().to('cuda')
    if color == chess.BLACK:
        x *= -1
    x = x.unsqueeze(0)
    move = predict(x)
    # print(move)
    vals = []
    froms = [str(legal_move)[:2] for legal_move in legal_moves]
    froms = list(set(froms))
    for from_ in froms:
        # print(move[0,:,:][0][0])
        val = move[0,:,:][0][8-int(from_[1]), letter_2_num[from_[0]]]
        # print(from_)
        vals.append(val)
    
    probs = distribution_over_moves(vals)

    chosen_from = str(np.random.choice(froms, size=1, p=probs)[0])[:2]

    vals = []
    for legal_move in legal_moves:
        from_ = str(legal_move)[:2]
        if from_ == chosen_from:
            to = str(legal_move)[2:]
            # print(move[0,:,:][0])
            # print(move[0,:,:][1])
            val = move[0,:,:][1][8 - int(to[1]), letter_2_num[to[0]]]
            vals.append(val)
        else:
            vals.append(0)
    chosen_move = legal_moves[np.argmax(vals)]
    # Create a new chess board
    board = chess.Board()

# Push a move to the board
    board.push_uci(chosen_move)
    b_mat = return_mat(board)
    return b_mat