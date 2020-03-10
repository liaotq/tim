from arch import arch_model
import numpy as np
import math


def predict_var(stock_returns, period):
    returns = np.array(list(stock_returns.values)).T
    initial_numberof_returns = returns[1].size
    esubt = []
    q_list = []
    # initial values
    cov_matrix = np.cov(returns)
    diag = cov_matrix.diagonal()
    diag_matrix = np.diag(np.diag(cov_matrix))
    garch11 = arch_model(diag, p=1, q=1)
    res = garch11.fit(update_freq=10)
    alpha = res.params[2]
    beta = res.params[3]

    esubt.append(np.matmul(np.linalg.pinv(diag_matrix), returns))
    q_list.append(np.ones((stock_returns.shape[1],stock_returns.shape[1])))

    pred_cov = []
    new_returns = []

    # garch 1,1
    def simulate_GARCH(T, a0, a1, b1, sigma1):
        # Initialize our values
        X = np.ndarray(T)
        sigma = np.ndarray(T)
        sigma[0] = sigma1

        for t in range(1, T):
            # Draw the next x_t
            X[t - 1] = sigma[t - 1] * np.random.normal(0, 1)
            # Draw the next sigma_t
            sigma[t] = math.sqrt(a0 + b1 * sigma[t - 1] ** 2 + a1 * X[t - 1] ** 2)

        X[T - 1] = sigma[T - 1] * np.random.normal(0, 1)

        return X, sigma

    # DCC Garch
    def dccGARCH(returns):
        cov_matrix = np.cov(returns)
        diag = cov_matrix.diagonal()
        diag_matrix = np.diag(np.diag(cov_matrix))

        garch11 = arch_model(diag, p=1, q=1)
        res = garch11.fit(update_freq=10)
        alpha = res.params[2]
        beta = res.params[3]

        esubt.append(np.matmul(np.linalg.pinv(diag_matrix), returns))
        qdash = np.cov(esubt[-2])
        q_list.append(((1 - alpha - beta) * qdash) +
                      (alpha * np.matmul(esubt[-2], esubt[-2].T)) + ((beta) * q_list[-1]))

        qstar = np.diag(np.diag(cov_matrix))
        qstar = np.sqrt(qstar)

        r = np.matmul(np.linalg.pinv(qstar), q_list[-1], np.linalg.pinv(qstar))

        pred_cov_temp = np.matmul(diag_matrix, r)
        pred_cov.append(np.matmul(pred_cov_temp, diag_matrix))

    periods_to_predict = period

    for n in returns:
        garch11 = arch_model(n, p=1, q=1)
        res = garch11.fit(update_freq=10)
        omega = res.params[1]
        alpha = res.params[2]
        beta = res.params[3]
        sig_ma = np.std(n)
        return_forecast, sigma_forecast = simulate_GARCH(periods_to_predict, omega, alpha, beta, sig_ma)
        n = np.append(n, return_forecast)
        new_returns.append(n)

    for i in range(initial_numberof_returns, new_returns[1].size):
        dccGARCH(returns[:, 0:i])

    cov = pred_cov[-1]

    return cov