'''
Classes that take in a scaler and return a residual and gradient
vector for minimisation for different scaling parameterisations.
'''
from dials.array_family import flex
from dials_scaling_helpers_ext import row_multiply
from scitbx import sparse
import abc

class ScalingTarget(object):
  
  __metaclass__  = abc.ABCMeta
  _grad_names = ['dI_dp']
  rmsd_names = ["RMSD_I"]
  rmsd_units = ["a.u"]
  
  def __init__(self, scaler, apm):
    self.scaler = scaler
    self.apm = apm
    self.weights = self.scaler.Ih_table.weights

    # Quantities to cache each step
    self._rmsds = None

  def predict(self):#do basis function calcuation and update Ih
    self.scaler.update_for_minimisation(self.apm)

  def get_num_matches(self):
    return self.scaler.Ih_table.size

  def rmsds(self):
    """calculate unweighted RMSDs for the matches"""
    # cache rmsd calculation for achieved test
    self._rmsds = [(flex.sum(self.calculate_residuals()**2)/self.scaler.Ih_table.size)**0.5]
    #print("rmsds %s" % self._rmsds)
    return self._rmsds

  def achieved(self):
    return False

  def calculate_residuals(self):
    '''returns a residual vector'''
    Ih_tab = self.scaler.Ih_table
    R = ((((Ih_tab.intensities - (Ih_tab.inverse_scale_factors * Ih_tab.Ih_values))**2)
          * Ih_tab.weights))
    return R

  def calculate_gradients(self):
    '''returns a gradient vector on length len(self.apm.x)'''
    Ih_tab = self.scaler.Ih_table
    gsq = ((Ih_tab.inverse_scale_factors)**2) * Ih_tab.weights
    sumgsq = gsq * Ih_tab.h_index_matrix
    rhl = Ih_tab.intensities - (Ih_tab.Ih_values * Ih_tab.inverse_scale_factors)
    dIh = ((Ih_tab.intensities - (Ih_tab.Ih_values * 2.0 * Ih_tab.inverse_scale_factors))
           * Ih_tab.weights)
    dIh_g = row_multiply(self.apm.derivatives, dIh)
    dIh_red = dIh_g.transpose() * Ih_tab.h_index_matrix
    dIh_by_dpi = row_multiply(dIh_red.transpose(), 1.0/sumgsq)
    term_1 = (-2.0 * rhl * Ih_tab.weights * Ih_tab.Ih_values *
              self.apm.derivatives)
    term_2 = (-2.0 * rhl * Ih_tab.weights * Ih_tab.inverse_scale_factors *
              Ih_tab.h_index_matrix) * dIh_by_dpi
    gradient = term_1 + term_2
    #if 'absorption' in self.apm.components_list:
    #  gradient += self.scaler.calc_absorption_constraint(self.apm)[1]
    return gradient

  def calculate_jacobian(self):
    '''returns a jacobian matrix of size Ih_table.size by len(self.apm.x)'''
    Ih_tab = self.scaler.Ih_table
    gsq = ((Ih_tab.inverse_scale_factors)**2)#  * Ih_tab.weights
    sumgsq = gsq * Ih_tab.h_index_matrix
    #rhl = Ih_tab.intensities - (Ih_tab.Ih_values * Ih_tab.inverse_scale_factors)
    dIh = ((Ih_tab.intensities - (Ih_tab.Ih_values * 2.0 * Ih_tab.inverse_scale_factors)))
    #  * Ih_tab.weights)
    dIh_g = row_multiply(self.apm.derivatives, dIh)
    dIh_red = dIh_g.transpose() * Ih_tab.h_index_matrix
    dIh_by_dpi = row_multiply(dIh_red.transpose(), 1.0/sumgsq)
    dIh_by_dpi = dIh_by_dpi.transpose() * Ih_tab.h_expand_matrix
    #now need to expand sum back out before row multiply
    term1 = row_multiply(dIh_by_dpi.transpose(), Ih_tab.inverse_scale_factors)
    term2 = row_multiply(self.apm.derivatives, Ih_tab.Ih_values)
    jacobian = sparse.matrix(term1.n_rows, term1.n_cols)
    for i, (col1, col2) in enumerate(zip(term1.cols(), term2.cols())):
      tot = col1 + col2
      jacobian[:, i] = tot
    return jacobian

  'methods for adaptlbfgs'
  def compute_functional_gradients(self):
    return flex.sum(self.calculate_residuals()), self.calculate_gradients()

  def compute_functional_gradients_and_curvatures(self):
    return flex.sum(self.calculate_residuals()), self.calculate_gradients(), None #curvatures

  def compute_restraints_functional_gradients_and_curvatures(self):
    'calculate restraints on parameters'
    restraints = None
    if 'absorption' in self.apm.components_list:
      restr = self.scaler.calc_absorption_constraint(self.apm)
      resid_restr = flex.sum(restr[0])
      grad_restr = restr[1]
      restraints = [resid_restr, grad_restr, None]
    return restraints #list of restraints to add to resid, grads and curvs?

  'methods for adaptlstbx (GN/ LM algorithms)'
  def compute_residuals(self): #for adaptlstbx (GN/ LM)
    return self.calculate_residuals(), self.weights

  def compute_residuals_and_gradients(self): #for adaptlstbx (GN/ LM)
    return self.calculate_residuals(), self.calculate_jacobian(), self.weights

  #def compute_restraints_residuals_and_gradients(self): #for adaptlstbx (GN/ LM)
  #  return restraints #list of restraints to add to residuals, jacobian and weights?


class target_function(object):
  '''Class that takes in a scaler and returns a residual
  and gradient function for minimisation.'''
  def __init__(self, scaler, apm):
    self.scaler = scaler
    self.apm = apm

  def calculate_residual(self):
    '''returns a residual vector'''
    Ih_tab = self.scaler.Ih_table
    R = ((((Ih_tab.intensities - (Ih_tab.inverse_scale_factors * Ih_tab.Ih_values))**2)
          * Ih_tab.weights))
    if 'absorption' in self.apm.components_list:
      R.extend(self.scaler.calc_absorption_constraint(self.apm)[0]) #FIX TO ALLOW GN minim.
    return R

  def calculate_gradient(self):
    '''returns a gradient vector'''
    Ih_tab = self.scaler.Ih_table
    gsq = ((Ih_tab.inverse_scale_factors)**2) * Ih_tab.weights
    sumgsq = gsq * Ih_tab.h_index_matrix
    rhl = Ih_tab.intensities - (Ih_tab.Ih_values * Ih_tab.inverse_scale_factors)
    dIh = ((Ih_tab.intensities - (Ih_tab.Ih_values * 2.0 * Ih_tab.inverse_scale_factors))
           * Ih_tab.weights)
    dIh_g = row_multiply(self.apm.derivatives, dIh)
    dIh_red = dIh_g.transpose() * Ih_tab.h_index_matrix
    dIh_by_dpi = row_multiply(dIh_red.transpose(), 1.0/sumgsq)
    term_1 = (-2.0 * rhl * Ih_tab.weights * Ih_tab.Ih_values *
              self.apm.derivatives)
    term_2 = (-2.0 * rhl * Ih_tab.weights * Ih_tab.inverse_scale_factors *
              Ih_tab.h_index_matrix) * dIh_by_dpi
    gradient = term_1 + term_2
    if 'absorption' in self.apm.components_list:
      gradient += self.scaler.calc_absorption_constraint(self.apm)[1]
    return gradient

  def calculate_residual_2(self):
    '''returns a residual vector'''
    Ih_tab = self.scaler.Ih_table
    R = ((((Ih_tab.intensities - (Ih_tab.inverse_scale_factors * Ih_tab.Ih_values))**2)
          * Ih_tab.weights))
    #if 'absorption' in self.apm.components_list:
    #  R.extend(self.scaler.calc_absorption_constraint(self.apm)[0]) #FIX TO ALLOW GN minim.
    return R

  def calculate_jacobian(self):
    '''returns a jacobian matrix'''
    Ih_tab = self.scaler.Ih_table
    gsq = ((Ih_tab.inverse_scale_factors)**2)#  * Ih_tab.weights
    sumgsq = gsq * Ih_tab.h_index_matrix
    #rhl = Ih_tab.intensities - (Ih_tab.Ih_values * Ih_tab.inverse_scale_factors)
    dIh = ((Ih_tab.intensities - (Ih_tab.Ih_values * 2.0 * Ih_tab.inverse_scale_factors)))
    #  * Ih_tab.weights)
    dIh_g = row_multiply(self.apm.derivatives, dIh)
    dIh_red = dIh_g.transpose() * Ih_tab.h_index_matrix
    dIh_by_dpi = row_multiply(dIh_red.transpose(), 1.0/sumgsq)
    dIh_by_dpi = dIh_by_dpi.transpose() * Ih_tab.h_expand_matrix
    #now need to expand sum back out before row multiply
    term1 = row_multiply(dIh_by_dpi.transpose(), Ih_tab.inverse_scale_factors)
    term2 = row_multiply(self.apm.derivatives, Ih_tab.Ih_values)
    jacobian = sparse.matrix(term1.n_rows, term1.n_cols)
    for i, (col1, col2) in enumerate(zip(term1.cols(), term2.cols())):
      tot = col1 + col2
      jacobian[:, i] = tot
    return jacobian

  def return_targets(self):
    '''return residual and gradient arrays'''
    return flex.sum(self.calculate_residual()), self.calculate_gradient()

  def return_residuals_jacobian_weight(self):
    'returns residual vector, jacobian and weights'
    return self.calculate_residual_2(), self.calculate_jacobian(), self.scaler.Ih_table.weights

class target_function_fixedIh(target_function):
  '''subclass to calculate the gradient for scaling against a fixed Ih'''
  def calculate_gradient(self):
    Ih_tab = self.scaler.Ih_table
    rhl = Ih_tab.intensities - (Ih_tab.Ih_values * Ih_tab.inverse_scale_factors)
    gradient = (-2.0 * rhl * Ih_tab.weights * Ih_tab.Ih_values * self.apm.derivatives)
    return gradient

"""class xds_target_function_log(target_function):
  '''Subclass that takes a data manager object and returns a residual and
  gradient function for a xds-like scaling parameterisation.'''
  def calculate_gradient(self):
    gradient = flex.double([])
    intensities = self.scaler.Ih_table.intensities
    scale_factors = self.scaler.Ih_table.inverse_scale_factors
    Ih_values = self.scaler.Ih_table.Ih_values
    scaleweights = self.scaler.Ih_table.weights
    gsq = ((scale_factors)**2) *scaleweights
    sumgsq = np.add.reduceat(gsq, self.scaler.Ih_table.h_index_cumulative_array[:-1])
    #sumgsq = flex.double(np.repeat(sumgsq, self.scaler.h_index_counter_array))

    rhl = intensities - (Ih_values * scale_factors)
    num = len(intensities)
    dIh = ((scale_factors * intensities) - (Ih_values * 2.0 * scale_factors)) * scaleweights
    for i in range(self.scaler.n_active_params):
      dIh_g = (dIh * self.apm.derivatives[i*num:(i+1)*num])
      dIh_g = np.add.reduceat(dIh_g, self.scaler.Ih_table.h_index_cumulative_array[:-1])/sumgsq
      dIh_g = flex.double(np.repeat(dIh_g, self.scaler.Ih_table.h_index_counter_array))
      drdp = -((Ih_values + dIh_g) * scale_factors
                * self.apm.derivatives[i*num:(i+1)*num])
      grad = (2.0 * rhl * scaleweights * drdp)
      gradient.append(flex.sum(grad))
    return gradient"""
