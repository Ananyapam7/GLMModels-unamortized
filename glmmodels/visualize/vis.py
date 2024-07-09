def viz_symm_matrix_history(self, param, name) -> None:
    '''Visualize the history of a symmetric matrix'''
    labels_dict = []
    with plt.style.context(plt.style.available[26]):
        fig, ax = plt.subplots()
        param = np.array(param)
        for k in range(param.shape[1]):
            for j in range(k, param.shape[2]):
                labels_dict.append(f"$\Sigma_{{{j}{k}}}$")
                ax.plot(param[:, k, j], label=f"$\Sigma_{{{j}{k}}}$")
        ax.grid(True)
        plt.legend(labels_dict, bbox_to_anchor=(0.0, 1.0), loc='upper left')
        plt.tight_layout()
        plt.xlabel("Iteration", fontsize=8)
        plt.savefig(f'illustration_results/{self.start_time}/{name}.pdf')
        
        plt.clf()

def viz_mat_history(self, param, name) -> None:
    '''Visualize the history of a matrix'''
    with plt.style.context(plt.style.available[26]):
        
        param = np.array(param)
        fig, ax = plt.subplots()
        for k in range(param.shape[1]):
            for j in range(param.shape[2]):
                ax.plot(param[:, k, j], label=f"{name}[{j},{k}]")
        ax.grid(True)
        plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
        plt.tight_layout()
        plt.xlabel("Iteration", fontsize=8)
        plt.savefig(f'illustration_results/{self.start_time}/{name}.png')
        
        plt.clf()

def viz_vec_history(self, param, name) -> None:
    '''Visualize the history of a vector'''
    with plt.style.context(plt.style.available[26]):
    
        param = np.array(param)
        
        fig, ax = plt.subplots()

        for k in range(param.shape[1]):
            ax.plot(param[:, k], label=f"$\mu_{k}$")
        ax.grid(True)
        plt.legend(bbox_to_anchor=(0.8, 0.8), loc='upper left')
        plt.tight_layout()
        plt.xlabel("Iteration", fontsize=8)
        plt.ylabel("$\mu$", fontsize=10)
        plt.savefig(f'illustration_results/{self.start_time}/{name}.pdf')
        plt.clf()

def viz_var_params(self) -> None:
    '''Visualize the history of the variational parameters'''
    # self.viz_vec_history(self.phi1_history, 'phi1')
    # self.viz_vec_history(self.phi2_history, 'phi2')
    # self.viz_vec_history(self.phi3_history, 'phi3')
    # self.viz_vec_history(self.phi4_history, 'phi4')
    self.viz_vec_norm_history([self.phi1_history, self.phi2_history, self.phi3_history, self.phi4_history])

def viz_model_params(self) -> None:
    '''Visualize the history of the model'''
    self.viz_vec_history(self.mu_history, 'mu')
    # self.viz_mat_history(self.L_history, 'L')
    # self.viz_vec_history(self.D_history, 'D')
    self.viz_symm_matrix_history(self.sigma_history, 'sigma')
    self.viz_mat_history(self.B_history, 'B')

def viz_vec_norm_history(self, params) -> None:
    '''Visualize the history of the norm of a vector'''    
    # for param in params:
    #     param = np.array(param)
    #     norm_val = np.linalg.norm(param, axis=1, ord=self.d)
    fig, ax = plt.subplots()
    # cmap = plt.cm.coolwarm
    # custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
    #         Line2D([0], [0], color=cmap(.3), lw=4),
    #         Line2D([0], [0], color=cmap(.6), lw=4),
    #         Line2D([0], [0], color=cmap(1.), lw=4)]
    ax.grid(True)
    
    for param in params:
        param = np.array(param)
        ax.plot(np.linalg.norm(param, axis=1, ord=self.d))
        # plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    # plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    plt.xlabel('Iteration')
    plt.ylabel(f'$\phi$\'s')
    ax.legend(['$\phi_1$', '$\phi_2$', '$\phi_3$', '$\phi_4$'], loc='upper right', fontsize=8, bbox_to_anchor=(1.0, 0.8))
    plt.savefig(f'illustration_results/{self.start_time}/phi_norm.pdf')
    plt.clf()


def viz_elbo(self) -> None:
    '''Visualize the ELBO'''
    fig, ax = plt.subplots()
    ax.plot(np.array(self.elbo_history))
    ax.grid(True)
    plt.xlabel('Iteration')
    plt.ylabel('ELBO')
    # plt.savefig(f'results/{self.folder_name}/{self.error}/{self.file_name}_{self.start_time}/elbo.png')
    plt.savefig(f'illustration_results/{self.start_time}/elbo.pdf')
    plt.clf()

def viz_convergence(self) -> None:
    '''Visualize the convergence of the variational parameters'''
    # self.viz_var_params()
    self.viz_model_params()
    self.viz_elbo()

def visualize_accuracy_metrics(self, mu, L, D, sigma, B) -> None:
    '''Visualize the accuracy metrics'''
    # plt.imshow(abs(self.sigma.numpy() - sigma.numpy()) , cmap = 'autumn' , interpolation = 'nearest')
    ax = sns.heatmap(abs(self.sigma.numpy() - sigma.numpy()) , linewidth = 0.5 , cmap = 'coolwarm' )
    plt.title("sigma")
    plt.savefig(f'results/{self.folder_name}/{self.error}/{self.file_name}_{self.start_time}/sigma_acc.png')
    plt.clf()

    fig, ax = plt.subplots()
    sns.heatmap([abs(self.mu.numpy() - mu.numpy())], linewidth = 0.5 , cmap = 'coolwarm' , ax = ax)
    plt.title("mu")
    plt.savefig(f'results/{self.folder_name}/{self.error}/{self.file_name}_{self.start_time}/mu_acc.png')
    plt.clf()