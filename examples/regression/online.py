"""
This script performs progressive regression for randomly generated data.
In contrast to "offline.py", it computes density progressively, using the
'run' method of PANENE.

To see the result, run this script with Python 3, Flask, and Flask-SocketIO
(we recommend to use Anaconda 3). Then, open 'localhost:8001' on your browser.
"""

from flask import Flask, render_template, request, send_from_directory
from flask_socketio import SocketIO, emit
from knnreg import KNNRegressor
import numpy as np

app = Flask(__name__, static_folder='')
app.config['SECRET_KEY'] = 'your secret key'
socketio = SocketIO(app)

N = 30000
X_space = np.linspace(0, np.pi * 2, N)
np.random.shuffle(X_space)
X = np.expand_dims(X_space, axis = 1).astype(np.float32)
y = np.sin(X).astype(np.float32)

X += np.random.normal(0, 0.1, X.shape)
y += np.random.normal(0, 0.1, y.shape)

sampleN = 100
samples = np.expand_dims(np.linspace(0, np.pi * 2, sampleN), axis = 1).astype(np.float32)

n_neighbors = 20

ops = 500

runner = None # A standard Thread object that run density estimation

def runner_job():
    neigh = KNNRegressor(X, y, n_neighbors=n_neighbors, online=True)

    inserted = 0

    while inserted < len(X):
        res = neigh.run(ops)
        inserted = res['numPointsInserted']

        y_pred = neigh.predict(samples)
         
        socketio.emit('result', {
            'points': [
                (x, y) for x, y in zip(X.reshape(N).tolist(), y.reshape(N).tolist())
            ][:500],
            'sampleN': sampleN,
            'samples': [
                (sample, score) for sample, score in zip(samples.reshape(sampleN).tolist(), y_pred.reshape(sampleN).tolist())
                ],
            'inserted': inserted,
            'total': len(X)
        })
        socketio.sleep(0.1)

def stop_runner():
    global runner

    if runner is not None:
        runner.kill()
    
    runner = None

@app.route('/')
def main():
    return app.send_static_file('online_visualizer.html')

@socketio.on('start')
def start():
    stop_runner()

    global runner
    if runner is None:
        runner = socketio.start_background_task(target=runner_job)

@socketio.on('stop')
def stop():
    stop_runner()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8001, debug=True)
