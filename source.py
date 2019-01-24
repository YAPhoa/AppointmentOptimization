import pandas as pd
import numpy as np
import math
import warnings
from functools import partial
from bayes_opt import BayesianOptimization

warnings.simplefilter("ignore")

np.random.seed(5)

rand_unif = np.random.uniform

#   simulator Function.
#   Each simulation should is generated daily

def simulate(service_rate = 3 , schedule_interval = 0.5 , opening_hours = 8 , n_days = 100 , no_show_prob = 0.0) :
    
    columns = [
        'Inter-arrival Time',
        'Arrival Time per Day',
        'Show up' ,
        'Service Starting Time',
        'Service Time',
        'Service Ending Time',
        'Waiting Time',
        'Sojourn Time',
        'Doctors Idle Time',
        'Number of People in the System',
        'Number of People Waiting',
    ]    
    
    cust_per_day = math.floor(opening_hours/schedule_interval)
    df_dict = {col : list() for col in columns}
    
    for i in range(n_days) :
        #### First Customer of the day
        inter_time = [schedule_interval]
        arrive_time = [schedule_interval]
        
        ###if random number bigger than no show probability mark as show up
        show_up = [1 if rand_unif() > no_show_prob else 0]
        
        service_s_time = [schedule_interval]
                    
        ##If show up do random roll for service time otherwise 0
        service_time = [-np.log(1-rand_unif())/service_rate if show_up[0] else 0]
        
        service_n_time = [service_s_time[0] + service_time[0]]
        wait_time = [service_s_time[0] - arrive_time[0]]
        
        sojourn_time = [service_n_time[0] - arrive_time[0]]
        doctors_idle = [0]
        n_people_sys = [1]
        n_people_wait = [max([n_people_sys[0] - 1 , 0])]
        
        for i in range(1, cust_per_day) :
            inter_time.append(schedule_interval)
            arrive_time.append(schedule_interval + arrive_time[i-1])
            
            if rand_unif() > no_show_prob :
                show_up.append(1)
            else :
                show_up.append(0)
                
            service_s_time.append(max([service_n_time[i-1] , arrive_time[i]])) 

            if show_up[i] :
                service_time.append(-np.log(1-rand_unif())/service_rate)
            else :
                service_time.append(0)
                
            service_n_time.append(service_s_time[i] + service_time[i])
            
            if show_up[i] :
                wait_time.append(service_s_time[i] - arrive_time[i])
            else :
                wait_time.append(0)
            sojourn_time.append(service_n_time[i] - arrive_time[i])
            
            doctors_idle.append(max([0 , arrive_time[i] - service_n_time[i-1]]))
            n_people_sys.append(sum((np.array(service_n_time[:i]) > arrive_time[i])*show_up[:i]) + show_up[i])
            n_people_wait.append(max([n_people_sys[i] - 1, 0]))
        
        df_dict['Inter-arrival Time'].extend(inter_time)
        df_dict['Arrival Time per Day'].extend(arrive_time)
        df_dict['Show up'].extend(show_up)
        df_dict['Service Starting Time'].extend(service_s_time)
        df_dict['Service Time'].extend(service_time)
        df_dict['Service Ending Time'].extend(service_n_time)
        df_dict['Waiting Time'].extend(wait_time)
        df_dict['Sojourn Time'].extend(sojourn_time)
        df_dict['Doctors Idle Time'].extend(doctors_idle)
        df_dict['Number of People in the System'].extend(n_people_sys)
        df_dict['Number of People Waiting'].extend(n_people_wait)
        
    return pd.DataFrame(df_dict)
    
def income(show_up , fee = 500) :
#   Calculate the income based on the number of patient who show up.
    return fee * show_up.sum()

def operation_cost(over_time , over_time_multiplier ,  hiring_cost, op_hour , op_h_cost = 200) :
    over_time_cost = (over_time*over_time_multiplier).sum() 
    op_cost = 0
    for op in op_hour :
        op_cost += op_h_cost * op
    return over_time_cost + op_cost + hiring_cost

def simulate_scheme(service_rate , schedule_interval , opening_hours ,
                    op_h_cost = 200 , no_show_prob = 0.2 , fee = 400 , n_days = 60) :

    #### Simulate a scheme with given configuration of service rate schedule interval and opening hour.
    #### This simulation will return the profit for a given simulated period.
        
    #### op_h_cost : Hourly Operational Cost for the clinic
    #### no_show_prob : Number of patient not showing up , this patient are not chargeable.
    #### n_days : Number of days every simulation is evaluated. Higher n_days will probably yield more stable result.        
    #### hiring_cost : The cost for doctor.     
    #### *Assumption : Doctor with higher skill will serve faster hence able to handle more patients.
    
    #### Define Cost for doctor's skill, the doctor is paid daily cost is scaled according to his work hour, 
    #### using 8 as base working hour
    hiring_cost = (1000 + 1800*(service_rate))*(opening_hours/8)*n_days

    #### Assume the doctor is paid for 1/5 of his wage if he is doing overtime    
    over_time_multiplier = hiring_cost/(n_days*5.0)
    
    #### simulate the data for n_days 
    ref = simulate(service_rate , schedule_interval , opening_hours , n_days = n_days , no_show_prob = no_show_prob)
    
    #### How long the doctor does overtime. Values can be negative if he leave earlier 
    #### than his designated work hour ends
    ot_count = (ref[ref['Arrival Time per Day'] == ref['Arrival Time per Day'].max()]['Service Ending Time'] \
                - opening_hours - schedule_interval).values
    
    
    #### The function below implies that if the doctor finishes his service with the last patient 
    #### before his designated work hour ends, he can leave early. He is only entitled to overtime
    #### pays if he is past his work hour. This is that is often used in machine learning which is named
    #### rectified linear unit
    relu = lambda x : max(x,0)

    overtime = np.array(list(map(relu , ot_count)))
    op_hour = np.array(ot_count + opening_hours) 
    inc = income(ref['Show up'] , fee)
    cst = operation_cost(overtime , over_time_multiplier , hiring_cost , op_hour , op_h_cost)
    return inc - cst

print('Single Server Optimization ...')    
base_parameter = {
    'service_rate' : 3 ,
    'schedule_interval' : 0.5
}

optimized = {}
print()

for no_show in [0,0.2] :

    print('With %.3f no show probability.' % no_show )
    fixed_params = {
    'op_h_cost' : 300 ,
    'no_show_prob' : no_show,
    'fee' : 600 ,
    'n_days' : 30 ,
    'opening_hours' : 8
    }
    print()
    print('Base Parameters : ')
    for key in base_parameter.keys() :
        print('%s : %.3f' %(key ,base_parameter[key]))
    print()
    print('Fixed Parameters :')
    for key in fixed_params.keys() :
        print('%s : %.3f' % (key , fixed_params[key]))
    print()

   #### Simulate the scheme for 100 months to assess the clinic's profit.
    print('Before Optimization :')
    sim = simulate(**base_parameter , n_days = 30 , opening_hours = 8 , no_show_prob = no_show)
    if no_show > 0 :
        sim.to_csv('Unoptimized with Probability.csv' , index = False)
    else :
        sim.to_csv('Unoptimized without Probability.csv' , index = False)

    simulate_scheme_fixed = partial(simulate_scheme , **fixed_params)
    res = []
    for i in range(100) :
        res.append(simulate_scheme_fixed(**base_parameter))
    
    print('Monthly Profit Mean : %.3f' % np.mean(res))
    print('Monthly Profit Variance : %.3f' %  np.var(res))
    print('Monthly Profit std : %.3f' %  np.std(res))
    
    
    mean_service_time = sim[sim['Show up'] == 1]['Service Time'].mean()
    var_service_time = sim[sim['Show up'] == 1]['Service Time'].var()
    std_service_time = sim[sim['Show up'] == 1]['Service Time'].std()
    print()
    bo = BayesianOptimization(simulate_scheme_fixed , 
                              {'service_rate' : (1.0,5.0) ,
                         #      'opening_hours' : (6.0,10.0)},
                                'schedule_interval' : (0.2,1.0)},
                              random_state = 10 , verbose = 0)
    bo.maximize(init_points = 5  , n_iter = 50)

    optimized[no_show] = bo.res['max']['max_params']

    print('After Optimization :')
    print()
    print('Best Parameters :')
    for key in optimized[no_show].keys() :
        print('%s : %.3f' %(key , optimized[no_show][key]))
    print()
    res = []

    for i in range(100) :
        res.append(simulate_scheme_fixed(**optimized[no_show]))
    
    print('Monthly Profit Mean : %.3f' %  np.mean(res))
    print('Monthly Profit Variance : %.3f' % np.var(res))
    print('Monthly Profit std : %.3f' % np.std(res))

    avg_idle = sim['Doctors Idle Time'].mean()
    var_idle = sim['Doctors Idle Time'].var()
    std_idle = sim['Doctors Idle Time'].std()
    avg_waiting = sim['Waiting Time'].mean()
    var_waiting = sim['Waiting Time'].var()
    std_waiting = sim['Waiting Time'].std()
    print()
	
	#### Assess the queuing system performance in 30 days.
    print('Queueing performance :')
    sim = simulate(**optimized[no_show] , n_days = 30 , opening_hours = 8 , no_show_prob = no_show)
    maxval = sim[['Arrival Time per Day']].max().values[0]
    avg_overtime = np.mean([max(num , 0) for num in (sim[sim['Arrival Time per Day'] == maxval]['Service Ending Time'] - fixed_params['opening_hours'] - optimized[no_show]['schedule_interval']).values])
    var_overtime = np.var([max(num , 0) for num in (sim[sim['Arrival Time per Day'] == maxval]['Service Ending Time'] - fixed_params['opening_hours']  - optimized[no_show]['schedule_interval']).values])
    std_overtime = np.std([max(num , 0) for num in (sim[sim['Arrival Time per Day'] == maxval]['Service Ending Time'] - fixed_params['opening_hours']  - optimized[no_show]['schedule_interval']).values])
	
    print('Mean Service Time : %.3f' % mean_service_time)
    print('Variance Service Time : %.3f' % var_service_time)
    print('Standard Deviation Service Time :%.3f' % std_service_time)
    print()
    print('Average Waiting Time : %.3f' % avg_waiting)
    print('Variance Waiting Time : %.3f' % var_waiting)
    print('Standard Deviation Waiting Time : %.3f' % std_waiting)
    print()
    print('Average Idle Time : %.3f' % avg_idle)
    print('Variance Idle Time : %.3f'% var_idle)
    print('Standard Deviation Idle Time : %.3f'% std_idle)
    print()
    print('Average Overtime: %.3f ' % avg_overtime)
    print('Variance Overtime : %.3f' % var_overtime)
    print('Standard Deviation Overtime : %.3f' % std_overtime)
    print()
    print('Average Show up : %.3f' % sim['Show up'].mean())
    print('Variance Show up : %.3f' % sim['Show up'].var())
    print('Standard Deviation Show up : %.3f' % sim['Show up'].std())
    print('=======================================================================')
    if no_show > 0 :
        sim.to_csv('Optimized with No Show.csv' , index = False)
    else :
        sim.to_csv('Optimized without No Show.csv' , index = False)


#### Simulation with 2 doctors, the doctor is assumed to have different service rates

def simulate_2(service_rate_1 = 3 , service_rate_2 = 3, schedule_interval = 0.15 , opening_hours = 8 , n_days = 1 , no_show_prob = 0.0) :
    
    columns = [
        'Inter-arrival Time',
        'Arrival Time per Day',
        'Show up' ,
        'Assigned Server',
        'Service Starting Time',
        'Service Time',
        'Doctor 1 Service Ending Time',
        'Doctor 2 Service Ending Time',
        'Waiting Time',
        'Sojourn Time',
#        'Doctors 1 Idle Time',	#Not Implemented yet
#        'Doctors 2 Idle Time', #Not implemented yet
        'Number of People in the System',
        'Number of People Waiting',
    ]    
    service_rates = [service_rate_1 , service_rate_2]
    cust_per_day = math.floor(opening_hours/schedule_interval)
    df_dict = {col : list() for col in columns}
    
    for i in range(n_days) :
        #### First Customer of the day
        inter_time = [schedule_interval]
        arrive_time = [schedule_interval]
        
        ###if random number bigger than no show probability mark as show up
        show_up = [1 if rand_unif() > no_show_prob else 0]
        assigned = 0
        assigned_server = [assigned + 1]
        service_s_time = [schedule_interval]           
        ##If show up do random roll for service time otherwise 0
        service_time = [-np.log(1-rand_unif())/service_rates[assigned] if show_up[0] else 0]		
       
        service_n_time = [[service_s_time[0] + service_time[0]] , [schedule_interval]]       
        wait_time = [service_s_time[0] - arrive_time[0]]
        
        sojourn_time = [service_n_time[assigned][0] - arrive_time[0]]        
#        doctors_idle_1 = [0] Not properly implemented yet
#        doctors_idle_2 = [0] Not properly implemented yet

        n_people_sys = [1]
        n_people_wait = [max([n_people_sys[0] - 1 , 0])]
        
        for i in range(1, cust_per_day) :
            inter_time.append(schedule_interval)
            arrive_time.append(schedule_interval + arrive_time[i-1])
            
            if rand_unif() > no_show_prob :
                show_up.append(1)
            else :
                show_up.append(0)
        
            assigned = np.argmin(np.array(service_n_time)[:,-1])

            assigned_server.append(assigned + 1)
            service_s_time.append(max([min([service_n_time[0][i-1] , service_n_time[1][i-1]]) , arrive_time[i]])) 
            
            if show_up[i] :

                service_time.append(-np.log(1-rand_unif())/service_rates[assigned])
            else :
                service_time.append(0)
                
            service_n_time[assigned].append(service_s_time[i] + service_time[i])

            service_n_time[1 - assigned].append(service_n_time[1-assigned][-1])

#            doctors_idle_1.append(service_s_time[-1] - max([arrive_time[-2] , service_n_time[0][-1]])) Not properly implemented yet
#            doctors_idle_2.append(service_s_time[-1] - max([arrive_time[-2] , service_n_time[1][-1]])) Not properly implemented yet
            
            if show_up[i] :
                wait_time.append(service_s_time[i] - arrive_time[i])
            else :
                wait_time.append(0)

            sojourn_time.append(service_n_time[assigned][i] - arrive_time[i])
            
            ctr = 0
            for n in range(i-1) :
                if service_n_time[assigned_server[n] - 1][n] > arrive_time[i] :
                    ctr+= show_up[n]
            n_people_sys.append(ctr)
            n_people_wait.append(max([n_people_sys[i] - 1, 0]))
        df_dict['Inter-arrival Time'].extend(inter_time)
        df_dict['Arrival Time per Day'].extend(arrive_time)
        df_dict['Show up'].extend(show_up)
        df_dict['Assigned Server'].extend(assigned_server)
        df_dict['Service Starting Time'].extend(service_s_time)      
        df_dict['Service Time'].extend(service_time)
        df_dict['Doctor 1 Service Ending Time'].extend(service_n_time[0])
        df_dict['Doctor 2 Service Ending Time'].extend(service_n_time[1])
        df_dict['Waiting Time'].extend(wait_time)
        df_dict['Sojourn Time'].extend(sojourn_time)
 #       df_dict['Doctors 1 Idle Time'].extend(doctors_idle_1) Not properly implemented yet
 #       df_dict['Doctors 2 Idle Time'].extend(doctors_idle_2) Not properly implemented yet
        df_dict['Number of People in the System'].extend(n_people_sys)
        df_dict['Number of People Waiting'].extend(n_people_wait)

    return pd.DataFrame(df_dict)


def operation_cost_2(overtime_1 , overtime_multiplier_1 ,  overtime_2 , overtime_multiplier_2 , hiring_cost, op_hour , op_h_cost = 200) :
    overtime_cost = (overtime_1*overtime_multiplier_1).sum()  + (overtime_2*overtime_multiplier_2).sum()
    op_cost = 0
    for op in op_hour :
        multiplier = op_h_cost
        op_cost += multiplier * op
    return overtime_cost + op_cost + hiring_cost


def simulate_scheme_2(service_rate_1 , service_rate_2, schedule_interval , opening_hours ,
                    op_h_cost = 200 , no_show_prob = 0.2 , fee = 400 , n_days = 60) :

	#### Similar scheme as above but with 2 doctors.
   
	#### Define Cost for doctor's skill, the doctor is paid daily cost is scaled according to his work hour, 
	#### using 8 as base working hour

	#### The function below assumes linear relationship with the service rate for the cost.
    hiring_cost_1 = (1000 + 1800*(service_rate_1))*(opening_hours/8)*n_days
    hiring_cost_2 = (1000 + 1800*(service_rate_2))*(opening_hours/8)*n_days
    hiring_cost = hiring_cost_1 + hiring_cost_2
	#### Assume the doctor is paid for 1/5 of his daily wage if he is doing overtime    
    overtime_multiplier_1 = hiring_cost_1/(n_days*5.0)
    overtime_multiplier_2 = hiring_cost_2/(n_days*5.0)
    
    #### simulate the data for n_days 
    params = {
        'service_rate_1' : service_rate_1,
        'service_rate_2' : service_rate_2,
        'schedule_interval' : schedule_interval,
        'opening_hours' : opening_hours,
        'n_days' : n_days,
        'no_show_prob' : no_show_prob,
    }
    ref = simulate_2(**params)
    
    #### How long the doctor does overtime. Values can be negative if he leave earlier than his designated work hour ends
    ot_count = (ref[ref['Arrival Time per Day'] == ref['Arrival Time per Day'].max()][['Doctor 1 Service Ending Time', 'Doctor 2 Service Ending Time']] \
                - opening_hours - schedule_interval).values

    relu = lambda x : x * (x > 0)
    ot_count = relu(ot_count)    
    op_hour = np.array(np.max(ot_count , axis =1) + opening_hours) 
    inc = income(ref['Show up'] , fee)
    
    cost_params = {
        'overtime_1' : ot_count[:,0],
        'overtime_multiplier_1' : overtime_multiplier_1,
        'overtime_2' : ot_count[:,1],
        'overtime_multiplier_2' : overtime_multiplier_2,
        'hiring_cost' : hiring_cost,
        'op_hour' : op_hour ,
        'op_h_cost' : op_h_cost
    }    
    
    cst = operation_cost_2(**cost_params)
    return inc - cst

print('Two server optimization ...')

fixed_params = {
    'op_h_cost' : 300 ,
    'no_show_prob' : 0,
    'fee' : 600 ,
    'n_days' : 30 ,
    'opening_hours' : 8
}
simulate_scheme_fixed_2 = partial(simulate_scheme_2 , **fixed_params)

base_parameter = {
    'service_rate_1' : 3 ,
    'service_rate_2' : 3 ,
    'schedule_interval' : 0.5 ,
}
print('Base Parameters :') 
for key in base_parameter.keys() :
    print('%s : %.3f' %(key, base_parameter[key]))
print()
print('Fixed Parameters :')
for key in fixed_params :
    print('%s : %.3f' %(key, fixed_params[key]))

sim = simulate_2(**base_parameter , n_days = 30 , opening_hours = 8 , no_show_prob = no_show)
sim.to_csv('Unoptimized 2 Servers.csv')

res = []
for i in range(100) :
    res.append(simulate_scheme_fixed_2(**base_parameter))
print()
print('Before Optimization :')    
print('Monthly Profit Mean : %.3f' %  np.mean(res))
print('Monthly Profit Variance : %.3f' %  np.var(res))
print('Monthly Profit std : %.3f' %  np.std(res))
print()	
bo_2 = BayesianOptimization(simulate_scheme_fixed_2 , 
                          {'service_rate_1' : (2.0,6.0) ,
                           'service_rate_2' : (2.0,6.0) ,
                            'schedule_interval' : (0.1,1.0)},
                          random_state = 10,
                          verbose = 0)
bo_2.maximize(init_points = 10  , n_iter = 50)
print('After Optimization :')
optimized_2 = bo_2.res['max']['max_params']
print('Best Parameters :')
for key in optimized_2.keys() :
    print(key , ':' , optimized_2[key])
print()
res = []
for i in range(100) :
    res.append(simulate_scheme_fixed_2(**optimized_2))

print('Monthly Profit Mean : %.3f' %  np.mean(res))
print('Monthly Profit Variance : %.3f' %  np.var(res))
print('Monthly Profit std : %.3f' %  np.std(res))
sim = simulate_2(**optimized_2 , n_days = 30 , opening_hours = 8 , no_show_prob = no_show)
sim.to_csv('Optimized 2 Servers.csv' , index = False)