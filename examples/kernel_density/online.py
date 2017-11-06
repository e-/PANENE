from flask import Flask, render_template, request, send_from_directory
from flask_socketio import SocketIO, emit
from knnkde import KNNKernelDensity
import numpy as np

app = Flask(__name__, static_folder='')
app.config['SECRET_KEY'] = 'your secret key'
socketio = SocketIO(app)

# create random data
def mv(n, mean, cov):
    return np.random.multivariate_normal(mean, cov, size=(n)).astype(np.float32)

N = 300000
X = np.concatenate((
    mv(N, [0.1, 0.3], [[0.01, 0], [0, 0.09]]),
    mv(N, [0.7, 0.5], [[0.04, 0], [0, 0.01]]),
    mv(N, [-0.4, -0.3], [[0.09, 0.04], [0.04, 0.02]])
    ), axis=0)

np.random.shuffle(X)

sampleN = 30
samples = np.indices((sampleN + 1, sampleN + 1)).reshape(2, -1).T / sampleN * 3 - 1.5

ops = 500

runner = None # A standard Thread object that run density estimation

def runner_job():
    kde = KNNKernelDensity(X, online=True)
    inserted = 0

    while inserted < len(X):

        res = kde.run(ops)
        inserted = res['numPointsInserted']

        scores = kde.score_samples(samples.astype(np.float32), k=20)
         
        socketio.emit('result', {
            'points': X[:500].tolist(),
            'bins': sampleN,
            'inserted': inserted,
            'total': len(X),
            'samples': [
                (sample, score) for sample, score in zip(samples.tolist(), scores.tolist())
            ]
        })
        socketio.sleep(0.01)

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
