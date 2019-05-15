import numpy as np


def factors(x):
    assert len(x) == 11
    phi = dict()
    phi['a'] = np.array([x[0], 1 - x[0]])
    phi['ab'] = np.array([[x[1], 1 - x[1]],
                          [x[2], 1 - x[2]]])
    phi['ae'] = np.array([[x[3], 1 - x[3]],
                          [x[4], 1 - x[4]]])
    phi['bc'] = np.array([[x[5], 1 - x[5]],
                          [x[6], 1 - x[6]]])
    phi['ced'] = np.array([[[x[7], 1 - x[7]],
                            [x[8], 1 - x[8]]],
                           [[x[9], 1 - x[9]],
                            [x[10], 1 - x[10]]]])
    return phi


def initial_clique_potentials(phi):
    psi = dict()
    # TODO ...
    return psi


def messages(psi):
    delta = dict()
    # TODO ...
    return delta


def beliefs(psi, delta):
    beta, mu = dict(), dict()
    # TODO ...
    return beta, mu


def query1(beta, mu):
    # TODO ...
    return q1


def query2(beta, mu):
    # TODO ...
    return q2


def query3(beta, mu):
    # TODO ...
    return q3


def belief_propagation(phi):
    psi = initial_clique_potentials(phi)
    delta = messages(psi)
    beta, mu = beliefs(psi, delta)
    return beta, mu


def main():
    # Try BP for a given set of parameters
    x = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
    beta, mu = belief_propagation(factors(x))
    print(query1(beta, mu))
    print(query2(beta, mu))
    print(query3(beta, mu))


if __name__ == '__main__':
    main()
