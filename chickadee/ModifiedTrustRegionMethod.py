import logging
from pyomo.contrib.trustregion.TRF import TrustRegionSolver
from pyomo.opt import SolverFactory, check_optimal_termination
from pyomo.contrib.trustregion.interface import TRFInterface
from pyomo.contrib.trustregion.util import IterationLogger
from pyomo.contrib.trustregion.filter import Filter, FilterElement

logger = logging.getLogger('pyomo.contrib.modifiedtrustregion')

def trust_region_method(model,
                        decision_variables,
                        ext_fcn_surrogate_map_rule,
                        config,
                        ipopt_options):
    """
    The main driver of the Trust Region algorithm method.

    Parameters
    ----------
    model : ConcreteModel
        The user's model to be solved.
    degrees_of_freedom_variables : List of Vars
        User-supplied input. The user must provide a list of vars which
        are the degrees of freedom or decision variables within
        the model.
    ext_fcn_surrogate_map_rule : Function, optional
        In the 2020 Yoshio/Biegler paper, this is refered to as
        the basis function `b(w)`.
        This is the low-fidelity model with which to solve the original
        process model problem and which is integrated into the
        surrogate model.
        The default is 0 (i.e., no basis function rule.)
    config : ConfigDict
        This holds the solver and TRF-specific configuration options.
    ipopt_options : Dict
        The options to be passed to the IPOPT solver
    """

    # Initialize necessary TRF methods
    TRFLogger = IterationLogger()
    TRFilter = Filter()
    interface = ModifiedTRFInterface(model, decision_variables,
                             ext_fcn_surrogate_map_rule, config, ipopt_options)

    # Initialize the problem
    rebuildSM = False
    obj_val, feasibility = interface.initializeProblem()
    # Initialize first iteration feasibility/objective value to enable
    # termination check
    feasibility_k = feasibility
    obj_val_k = obj_val
    # Initialize step_norm_k to a bogus value to enable termination check
    step_norm_k = 0
    # Initialize trust region radius
    trust_radius = config.trust_radius

    iteration = 0

    TRFLogger.newIteration(iteration, feasibility_k, obj_val_k,
                           trust_radius, step_norm_k)
    TRFLogger.logIteration()
    if config.verbose:
        TRFLogger.printIteration()
    while iteration < config.maximum_iterations:
        iteration += 1

        # Check termination conditions
        if ((feasibility_k <= config.feasibility_termination)
            and (step_norm_k <= config.step_size_termination)):
            print('EXIT: Optimal solution found.')
            interface.model.display()
            break

        # If trust region very small and no progress is being made,
        # terminate. The following condition must hold for two
        # consecutive iterations.
        if ((trust_radius <= config.minimum_radius) and
            (abs(feasibility_k - feasibility) < config.feasibility_termination)):
            if subopt_flag:
                logger.warning('WARNING: Insufficient progress.')
                print('EXIT: Feasible solution found.')
                break
            else:
                subopt_flag = True
        else:
            # This condition holds for iteration 0, which will declare
            # the boolean subopt_flag
            subopt_flag = False

        # Set bounds to enforce the trust region
        interface.updateDecisionVariableBounds(trust_radius)
        # Generate suggorate model r_k(w)
        if rebuildSM:
            interface.updateSurrogateModel()

        # Solve the Trust Region Subproblem (TRSP)
        obj_val_k, step_norm_k, feasibility_k = interface.solveModel()

        TRFLogger.newIteration(iteration, feasibility_k, obj_val_k,
                               trust_radius, step_norm_k)

        # Check filter acceptance
        filterElement = FilterElement(obj_val_k, feasibility_k)
        if not TRFilter.isAcceptable(filterElement, config.maximum_feasibility):
            # Reject the step
            TRFLogger.iterrecord.rejected = True
            trust_radius = max(config.minimum_radius,
                               step_norm_k*config.radius_update_param_gamma_c)
            rebuildSM = False
            interface.rejectStep()
            # Log iteration information
            TRFLogger.logIteration()
            if config.verbose:
                TRFLogger.printIteration()
            continue

        # Switching condition: Eq. (7) in Yoshio/Biegler (2020)
        if ((obj_val - obj_val_k) >=
             (config.switch_condition_kappa_theta
             * pow(feasibility, config.switch_condition_gamma_s))
            and (feasibility <= config.minimum_feasibility)):
            # f-type step
            TRFLogger.iterrecord.fStep = True
            trust_radius = min(max(step_norm_k*config.radius_update_param_gamma_e,
                                   trust_radius),
                               config.maximum_radius)
        else:
            # theta-type step
            TRFLogger.iterrecord.thetaStep = True
            filterElement = FilterElement(obj_val_k - config.param_filter_gamma_f*feasibility_k,
                                          (1 - config.param_filter_gamma_theta)*feasibility_k)
            TRFilter.addToFilter(filterElement)
            # Calculate ratio: Eq. (10) in Yoshio/Biegler (2020)
            rho_k = ((feasibility - feasibility_k + config.feasibility_termination) /
                     max(feasibility, config.feasibility_termination))
            # Ratio tests: Eq. (8) in Yoshio/Biegler (2020)
            # If rho_k is between eta_1 and eta_2, trust radius stays same
            if ((rho_k < config.ratio_test_param_eta_1) or
                (feasibility > config.minimum_feasibility)):
                trust_radius = max(config.minimum_radius,
                                   (config.radius_update_param_gamma_c
                                   * step_norm_k))
            elif (rho_k >= config.ratio_test_param_eta_2):
                trust_radius = min(config.maximum_radius,
                                   max(trust_radius,
                                       (config.radius_update_param_gamma_e
                                       * step_norm_k)))

        TRFLogger.updateIteration(trustRadius=trust_radius)
        # Accept step and reset for next iteration
        rebuildSM = True
        feasibility = feasibility_k
        obj_val = obj_val_k
        # Log iteration information
        TRFLogger.logIteration()
        if config.verbose:
            TRFLogger.printIteration()

    if iteration >= config.maximum_iterations:
        logger.warning('EXIT: Maximum iterations reached: {}.'.format(config.maximum_iterations))

    return interface.model


@SolverFactory.register('modifiedtrustregion', doc='TrustRegion method modified to pass through IPOPT Options')
class ModifiedTrustRegionSolver(TrustRegionSolver):
    def __init__(self, **kwds):
        # Keep ipopt options from being included in self.config as it does not fit the type
        self.ipopt_options = kwds.pop('ipopt_options', {})
        super().__init__(**kwds)

    def solve(self, model, degrees_of_freedom_variables,
              ext_fcn_surrogate_map_rule=None, **kwds):
        """
        This method calls the TRF algorithm.

        Parameters
        ----------
        model: ``ConcreteModel``
            The model to be solved using the Trust Region Framework.
        degrees_of_freedom_variables : List of Vars
            User-supplied input. The user must provide a list of vars which
            are the degrees of freedom or decision variables within
            the model.
        ext_fcn_surrogate_map_rule : Function, optional
            In the 2020 Yoshio/Biegler paper, this is refered to as
            the basis function `b(w)`.
            This is the low-fidelity model with which to solve the original
            process model problem and which is integrated into the
            surrogate model.
            The default is 0 (i.e., no basis function rule.)

        """
        config = self.config(kwds.pop('options', {}))
        config.set_value(kwds)
        if ext_fcn_surrogate_map_rule is None:
            # If the user does not pass us a "basis" function,
            # we default to 0.
            ext_fcn_surrogate_map_rule = lambda comp,ef: 0
        result = trust_region_method(model,
                            degrees_of_freedom_variables,
                            ext_fcn_surrogate_map_rule,
                            config,
                            self.ipopt_options)
        return result

class ModifiedTRFInterface(TRFInterface):
    def __init__(self, model, decision_variables, ext_fcn_surrogate_map_rule, config, ipopt_options={}):
        super().__init__(model, decision_variables, ext_fcn_surrogate_map_rule, config)
        self.ipopt_options = ipopt_options

    def solveModel(self):
        """
        Call the specified solver to solve the problem.

        Returns
        -------
            self.data.objs[0] : Current objective value
            step_norm         : Current step size inf norm
            feasibility       : Current feasibility measure

        This also caches the previous values of the vars, just in case
        we need to access them later if a step is rejected
        """
        current_decision_values = self.getCurrentDecisionVariableValues()
        self.data.previous_model_state = self.getCurrentModelState()
        results = self.solver.solve(
            self.model, keepfiles=self.config.keepfiles, tee=self.config.tee, options=self.ipopt_options
        )

        if not check_optimal_termination(results):
            raise ArithmeticError(
                'EXIT: Model solve failed with status {} and termination'
                ' condition(s) {}.'.format(
                    str(results.solver.status),
                    str(results.solver.termination_condition),
                )
            )

        self.model.solutions.load_from(results)
        new_decision_values = self.getCurrentDecisionVariableValues()
        step_norm = self.calculateStepSizeInfNorm(
            current_decision_values, new_decision_values
        )
        feasibility = self.calculateFeasibility()
        return self.data.objs[0](), step_norm, feasibility