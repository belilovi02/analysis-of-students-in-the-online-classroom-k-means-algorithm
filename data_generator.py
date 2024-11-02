import numpy as np
import pandas as pd

def generate_student_data(num_students=500, filename='student_data.csv'):
    """
    Generiše simulirane podatke o studentima i čuva ih u CSV fajlu.

    :param num_students: Broj studenata za generisanje (default: 100)
    :param filename: Ime CSV fajla u koji će podaci biti sačuvani (default: 'student_data.csv')
    """
    np.random.seed(42)
    
    # Generisanje podataka
    attendance = np.random.randint(0, 21, num_students)  # Bodovi za prisustvo (0-20)
    
    # Generisanje ocena za zadatke (3 zadatka, po 10 bodova)
    task_scores = np.random.randint(0, 11, (num_students, 3))  # Bodovi za 3 zadatka
    total_task_scores = task_scores.sum(axis=1)  # Ukupni bodovi za zadatke (0-30)

    exam_scores = np.random.randint(0, 51, num_students)  # Bodovi za ispit (0-50)

    # Ukupni bodovi
    total_scores = attendance + total_task_scores + exam_scores  # Zbir bodova (0-100)

    data = pd.DataFrame({
        'Attendance': attendance,
        'Task_Scores': total_task_scores,
        'Exam_Score': exam_scores,
        'Total_Score': total_scores
    })

    data.to_csv(filename, index=False)
    print(f'Data saved to {filename}')
