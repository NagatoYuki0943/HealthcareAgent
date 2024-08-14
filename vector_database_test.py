import time
from vector_database import VectorDatabase
from loguru import logger
from tqdm import tqdm


DATA_PATH: str  = "./data"
EMBEDDING_MODEL_PATH: str  = "./models/bce-embedding-base_v1"
RERANKER_MODEL_PATH: str = "./models/bce-reranker-base_v1"
PERSIST_DIRECTORY: str  = "./vector_db/faiss"
SIMILARITY_TOP_K: int = 4
SIMILARITY_FETCH_K: int = 10
SCORE_THRESHOLD: float = 0.15
ALLOW_SUFFIX: tuple[str] = (".txt", ".md", ".docx", ".doc", ".pdf")
VECTOR_DEVICE = 'cuda'
TEXT_SPLITTER_TYPE = 'RecursiveCharacterTextSplitter'

vector_database = VectorDatabase(
    data_path = DATA_PATH,
    embedding_model_path = EMBEDDING_MODEL_PATH,
    reranker_model_path = RERANKER_MODEL_PATH,
    persist_directory = PERSIST_DIRECTORY,
    similarity_top_k = SIMILARITY_TOP_K,
    similarity_fetch_k = SIMILARITY_FETCH_K,
    score_threshold = SCORE_THRESHOLD,
    allow_suffix = ALLOW_SUFFIX,
    device = VECTOR_DEVICE,
    text_splitter_type = TEXT_SPLITTER_TYPE,
)
# 创建数据库
vector_database.create_faiss_vectordb(force=False)
# 载入数据库(创建数据库后不需要载入也可以)
vector_database.load_faiss_vectordb()
# 创建相似度 retriever
# vector_database.create_faiss_retriever()
# 创建重排序 retriever
vector_database.create_faiss_reranker_retriever()


querys = [
    'Intravenous EDTA Chelation Treatment of a Patient with',
    'Eye Pressure Lowering Effect of Vitamin C',
    'A Comparison of Oral Health Between Multiple Sclerosis',
    'High-Dose-Intravenous-Vitamin-C-Treatment-for-Zika-Fever-31.1',
    'Vitamin B3 and Krebiozen - a polemic',
    'The Parenteral Use of Vitamins in the Treatment',
    'A Vitamin B3 Dependent Family',
    'The Use of Mega Vitamin Therapy in Regulating Severe',
    'Vitamin B3 Dependent Child',
    'The Use of Vitamin B12b in Psychiatric Practice',
    'A Study of Neurological Organization Procedures',
    'Clinical Observations on the Treatment of Schizophrenic',
    'An Examination of the Double-Blind Method',
    'Early Evidence About Vitamin C And the Common Cold',
    'To the Editor The Road to Shangri-La is Paved with',
    'Megavitamins',
    'Administration of Massive Doses of Vitamin E',
    'The Vitamin D-Problem An Important Lesson',
    'Diet-Vitamin Program for Jail Inmates',
    'The Use of Megavitamin Treatment in Children',
    'An Update of Megavitamin Therapy in Orthomolecular',
    'The Eating Habits of High and Low Vitamin C Users',
    'Megavitamin Therapy for Different Cases',
    'Massive Vitamin C as an Adjunct in Methadone Maintenance',
    'Stomach Acid and Megavitamins',
    'A Report on a Twelve-Month Period of Treating Metabolic Diseases',
    'Does Ascorbic Acid Destroy Vitamin B12',
    'Meditation Protein Diet and Megavitamins in the Treatment',
    'Reduction of Blood Lead Levels in Battery Workers',
    'A Reply to the American Psychiatric Association Task',
    'Large Amounts of Nicotinic Acid and Vitamin B12',
    'X-Linked Dominant Manic-Depressive Illness Linkage with',
    'Resistance To Orthomolecular Medicine Or Why You Dont',
    'Orthomolecular Medicine and Megavitamin Therapy',
    'Vitamins B1 B6 and B12 In The Adjunctive Treatment',
    'Vitamins The Get-Smart Pills',
    'Dr Nolan DC Lewis 1889-1979',
    'The Method of Determining Proper Doses of Vitamin C',
    'Psychiatric Significance of the Plasma Concentrations',
    'Treatment of a Mucopolysaccharide Type of Storage',
    'Vitamins Bl B6 and B12 in the Adjunctive Treatment',
    'Vitamin B6 Nutritional Status of a Psychiatric',
    'Vitamin C and Tolerance of Heat and Cold Human Evidence',
    'Vitamin B15 A Review and Update',
    'Vitamin B-12 Levels of Cerebrospinal Fluid in Patients',
    'Alzheimers Disease Alcohol Dementia Association',
    'William H Lylc Jr',
    'Nutrient Pioneers Alva Rae Patton Conrad Elvehjem',
    'The Effect of EDTA Chelation Therapy With Multivitamin',
    'The Clinical Change in Patients Treated with EDTA',
    'National Institute of Health Promotes Megavitamin Therapy',
    'The Prevention Of Tardive Dyskinesia with High Dosage',
    'Vitamin Therapy for Hyperactivity and Schizophreniae',
    'The Ideal Vitamin C Intake',
    'Alzheimers Dementia Some Possible Mechanisms Related To Vitamins',
    'Im Schizophrenic Doctor Not Stupid',
    'Around The World International Vitamin Convention',
    'Correspondence',
    'Around The World AIDS Vitamin C and Egg Lecithin',
    'Ascorbic acid and mental depression',
    'Megavitamin Therapy in the Reduction of Anxiety',
    'Case study High Dose intravenous Vitamin C in the',
    'Nutritional Interrelationships Minerals Vitamins',
    'Hardin Jones Biostatistical Analysis of Mortality Data',
    'Cardiovascular Dynamics and Edta Chelation with',
    'The Nutritional Relationships of Vitamin A',
    'Treatment of Hypercholesterolemia with Vitamin E C',
    'The Origin of the 42-year Stonewall of Vitamin C',
    '2 Welcome To Second World Congress on Vitamin C',
    '7 Vitamin C and Stomatology A Mouthful of Evidence',
    '8 Clinical Procedures in Treating Terminally Ill Cancer',
    '9 Vitamin C and Multifactorial Disease',
    '0 Vitamin C Deficiency Cholesterol Metabolism',
    '2 Children Vitamin C and Medical Progress',
    'Vitamin Mineral Supplementation and the intelligence',
    'Margaret Jean Callbeck RN 1916-1992',
    'The Third Face of Vitamin C',
    'Cancer Immunology and Aging The Nutritional influence',
    'Megavitamins and Psychotherapy Effective Economical',
    'Hardin Jones Biostatistical Analysis of Mortality Data',
    'Vitamin C and Fatigue',
    'Vitamin B6 and Carpal Tunnel Syndrome A Case Report',
    'Pride Prejudice and Vitamin C',
    'Predictive Medicine The Story of Odds and Ends',
    'Discovering Chinese Mineral Drugs',
    'The Cytotoxic Food Sensitivity Test An Important',
    'Staged Management for Chronic Fatigue Syndrome',
    'The Violation of Childhood Toxic Metals and',
    'High Dose intravenous Vitamin C and Long Time Survival',
    'Antioxidants in Health and Disease The Big Picture',
    'Treatment of Iritis and Herpes Zoster with Vitamin C',
    'Both Feet Back on the Ground',
    'The Need To Liberate Physicians To Practice',
    'Agitation Allergies and Attention Deficit',
    'Theoretical Evidence That the Ebola Virus Zaire Strain',
    'Cretinism The Iodine-selenium Connection',
    'Mercury and Acrodynia',
    'Parkinsons Disease and Mercury',
    'Adverse Effects of Zinc Deficiency A Review from the',
    'Eye Pressure Lowering Effect of Vitamin C',
    'Orthomolecular The Optimum Treatment for Schizophrenia',
    'Minerals and Disease',
    'Coronary Artery Occlusion Chelation and Cholesterol',
    'Hair Trace Element Status of Appalachian Head Start',
    'Glycemic Modulation of Tumor Tolerance',
    'The Serotonin Connection',
    'Cranial Electrical Stimulation',
    'Vitamin C and Hot Flashes FACT Use in Chronic',
    'How the Sick Get Sicker by Following Current Medical',
    'The Adverse Effects of Manganese Deficiency',
    'Intravenous Vitamin C in A Terminal Cancer Patient',
    'Botanical Inhibitors of Amine Oxidase Relevance To',
    'Mercury Vapour in the Oral Cavity in Relation To the',
    'Unrecognized Pandemic Subclinical Diabetes of the',
    'Cancer Therapy by Immobilizing Mitotic Energy Sources',
    'Coenzyme Q10 and Cancer Chronic Sulfite Toxicity',
    'Orthomolecular Medicine in the Universe',
    'How To Live Longer and Feel Better–even with Cancer',
    'Surviving Unipolar Depression–the Effectiveness of',
    'Reduction of Cholesterol and LpA in Regression of',
    'Serotonin and Health',
    'Cobalamin Deficiency Methylation and Neurological',
    'Symptoms Before and After Proper Amalgam Removal',
    'Iliotibial Band Friction Syndrome',
    'Epstein–Barr Virus infections in Patients',
    'Coenzyme Q10 A Novel Cardiac Antioxidant',
    'Beta-carotene and Other Carotenoids Promises Failures',
    'Red Blood Cell Fatty Acids As A Diagnostic Tool',
    'Sick Building Syndrome',
    'Paraphilias As a Subtype of Obsessive-Compulsive',
    'Maitake D-fraction Healing and Preventive Potential for',
    'Nutritional Support and Prognosis of Patients with',
    'Myalgic Encephalomyelitis Me A Haemorheological',
    'Neurobiology and Quantified EEG of Coenzyme Q10',
    'Urinary Pyrrole in Health and Disease',
    'A Stone That Kills Two Birds How Pantothenic Acid',
    'The Epidemiological Structure of Multiple Sclerosis',
    'Antioxidants and Cancer A Brief Discussion on',
    'A Theoretical Biochemical Basis of Cancer',
    'Ascorbic Acid Effect on Plasma Amino Acids',
    'Popliteal Artery Entrapment Syndrome A Case Report',
    'Evidence that Mercury from Silver Dental Fillings may be',
    'The Effects of in vitro Electrical Stimulation on',
    'Nutritional and Lifestyle Modification To Augment',
    'Peroxisomal Disturbances in Autistic Spectrum Disorder',
    'The Patient with A Harmful Hobby and the the Depressed',
    'The Results from Red Cell Shape Analyses of Blood',
    'Selenium and Viral Diseases Facts and Hypotheses',
    'Evidence that Mercury from Dental Amalgam May Have',
    'Selenium and Cancer A Geographical Perspective',
    'Recent Advances in Oxidative Stress and Antioxidants',
    'Paradigms and Miracles Alternative Medicine',
    'Psychometric Evidence That Dental Amalgam Mercury May Be',
    'Successful Reversal of Retinitis Pigmentosa',
    'Alpha-Lipoic Acid Thioctic Acid My Experience with',
    'Lumbar Facet Syndrome A Case Report',
    'Folate and Neural Tube Defect Risk Paradigm Shift after',
    'Playing with Statistics Or Lies Damn Lies and Statistics',
    'High-dose intravenous Vitamin C in the Treatment of A',
    'Treatment of Hypertension from An Orthomolecular',
    'Functional Diagnosis in Nutritional Medicine',
    'The Bodys Negative Response to Excess Dietary Protein',
    'Reassessing the Role of Sugar in the Etiology of Heart',
    'Hemorrhagic Stroke in Human Pretreated with Coenzyme Q10',
    'The Clinical Use of Bovine Colostrum',
    'The Paradigm Wars Continue',
    'The Nature and Structure of Febrile Psychosis in the Sudan',
    'Another Reason for Change The Distinct Philosophies',
    'Evidence that Mercury from Silver Dental Fillings May',
    'Joint and Muscle Pain Various Arthritic Conditions and',
    'Calcium and Cancer A Geographical Perspective',
    'Regarding Double-Blind Medical Trials Foods as',
    'Observations On the Dose and Administration of Ascorbic',
    'Macular Degeneration Treatment with Nutrients',
    'Potassium A New Treatment for Premenstrual Syndrome',
    'The Health of the NaturopathVitamin Supplementation',
    'The Application of the Hardin Jones-Pauling',
    'The Liver Mechanisms of Toxic Injury',
    'Histamine Levels in Health and Disease',
    'Diagnosing Schizophrenia Past Present and Future',
    'Treatment of Ambulant Schizophrenics with Vitamin B3',
    'Zinc and Manganese in the Schizophrenias 1983',
    'The Adrenochrome Hypothesis and Psychiatry 1990',
    'Patentable vs Non-Patentable Treatment',
    'Alzheimers Disease An Unusual Story of Identical Twins',
    'Conventional and Unconventional Medical Practice',
    'Schizophrenia The Latex Allergy Hypothesis',
    'Intravenous EDTA Chelation Treatment of a Patient with',
    'A Comparison of Oral Health Between Multiple Sclerosis',
    'The Neurobiology of Lipids in Autistic Spectrum Disorder',
    'The Biochemical Treatment of Schizophrenia Revisited',
    'Message from the President',
    'Should Tranquilizers Be Used Preventively for',
    'Parkinson’s Disease Multiple Sclerosis and Amyotrophic',
    'Vaccinations Inoculations and Ascorbic Acid',
    'Review of Growth Hormone Therapy',
    'Reduced Risk from Parkinsons Disease in Smokers',
    'The Underlying Mechanisms of Brain Allergies',
    'Urine Indican as an Indicator of Disease',
    'A Brief Update on Ubiquinone',
    'The Negative Health Effects of Chlorine',
    'Lycopene Its Role in Health and Disease',
    'Niacin and Cholesterol The History of a Discovery',
    'Mental Illness and the Mind-Body Problem',
    'Introduction of Niacin as the First Successful Treatment',
    'Therapeutic Effect of d-Lenolate Against Experimental',
    'Vitamin C and Cancer - A Workshop',
    'Vitamin C as Cancer Therapy An Overview',
    'Vitamin C Case History of an Alternative Cancer Therapy',
    'Clinical Evaluation of Vitamin C and other',
    'Antioxidant Nutrients and Cancer',
    'Remission of Stage IV Metastatic Ocular Melanoma',
    'Parasitic Worm Beneficial in Inflammatory Bowel Disease',
    'Mineral Status Toxic Metal Exposure and Childrens',
    'The Role of Homocysteine in Human Health',
    'The Search For Vitamin C Toxicity',
    'Does Water Fluoridation have Negative Side Effects',
    'Cancer Lifestyle Modification and Glucarate',
    'The Influence of “Junk” Science in Manipulating',
    'The Skin in Health and Disease',
    'From a Violent 17-Year Old Male to a Normal Young Man',
    'Enhanced Resistance against Influenza Virus by Treatment',
    'Cases from the Periphery Selenium EPA and',
    'Recommendation of Herbal Remedies by Psychiatrists',
    'Red Blood Cell Shape Symptoms and Reportedly Helpful',
    'Vitamin C in Cardiovascular Disease',
    'The Hepatitis C Patient Functional Assessment and',
    'The Effect of Alternating Magnetic Field Exposure and',
    'Oralmat and HIV Disease A Report of Five Cases',
    'Red Blood Cell Shapes in Women with Fibromyalgia and the',
    'Thyroid and Schizophrenia',
    'Vitamin C Symptoms and Respiratory Symptoms',
    'Is Vitamin B3 Dependency a Causal Factor',
    'Mood Correlates of Substance Use Among Chronic Mentally',
    'The Role of Vitamins B3 and C in the Treatment',
    'Fatigue and Vitamin C',
    'Preventive Health Screening Program in an Industrial',
    'Soy Isoflavones and Breast Cancer',
    'Homeostasis A Continuous Search for Health',
    'Mercury Dental Amalgams The Controversy Continues',
    'Facial Effects of the Warner Protocol for Children with',
    'Case from the Center Sixteen-Year History with High',
    'Alzheimers Disease Pathogenesis The Role of Aging',
    'Factors in Neurotoxicity in Adolescents',
    'Assessment of Granulocyte Activity with Application',
    'Carbohydrate Consumption and Cardiovascular Complaints',
    'Shaken Baby Syndrome or Scurvy',
    'Detection of the Level of Energy Metabolism in Patients',
    'Natural Therapies for Reducing Intraocular Eye Pressure',
    'Vitamin C and Oxidative DNA Damage Revisited',
    'Allopathic Medicine’s Stonewall Shaken',
    'Detection of Energy Metabolism Level in Cancer Patients',
    'Caffeine Anaphylaxis A Progressive Toxic Dementia',
    'Taking the Cure The Pioneering Work of William Kaufman',
    'A Consideration of Vitamin B3 as an Inhibitor of the',
    'Urine Pyrroles in Patients with Cancer',
    'Vitamin E and Heart Disease Controversy Two Major',
    'Indices of Pyridoxine Levels on Symptoms Associated with',
    'Effect of Vitamin C Supplementation on Ex Vivo Immune',
    'Hair Lead and Cadmium Levels and Specific Depressive and',
    '1 Editorial Toxic Vitamins',
    '2 Safe Upper Limits for Nutritional Supplements',
    '3 Vitamin A and Beta-Carotene',
    '4 Negative and Positive Side Effects of Vitamin B3',
    '5 Vitamin B6 Extract of Submission to the UK’s Food',
    '6 A Comment on Safe Upper Levels of Folic Acid B6 and B12',
    '7 Side Effects of Over-the-Counter Drugs',
    '8 The Trials and Tribulations of Vitamin C',
    '9 The Gift of Vitamin C',
    '0 Vitamin D Deficiency Diversity and Dosage',
    '1 Vitamin E A Cure in Search of Recognition',
    '2 Can Vitamin Supplements Take the Place of a Bad Diet',
    'Hypochlorhydria and Multiple Organ Failure A Leading',
    'Taking the Cure The Pioneering Work of Ruth Flinn',
    'Vitamin D Supplementation in the Fight Against Multiple',
    'Vitamin B3 for Nicotine Addiction',
    'Xenoestrogens and Breast Cancer',
    'The Energy System Creating Life and Cancer from',
    'Seasonality of Birth in Alzheimers Disease',
    'Alzheimer’s Disease Minerals and Essential Fatty Acids',
    'The Use of Vitamin C and Other Antioxidants with',
    'The Use of Vitamin C with Chemotherapy in Cancer',
    'The Use of Antioxidants with Chemotherapy and',
    'Folic Acid Vitamin D and Prehistoric Polymorphisms',
    'Monitoring of ATP Levels in Red Blood Cells and T Cells',
    'The Benefits of Going Beyond Conventional Therapies',
    'Alzheimer’s Disease and Trace Elements Chromium and Zinc',
    'Optimal Dosing for Schizophrenia',
    'Vitamin C as an Ergogenic Aid',
    'Gingival Bleeding in Smoking and Non-smoking Subjects',
    'Vitamin C and Osteoporosis Is There a Connection',
    'Can One Vitamin Overcome the General Nutrient',
    'Ascorbic Acid and the Immune System',
    'The Effect of Nutritional Therapy for Yeast Infection',
    'The Effect of Neurosteroids on Depression in Peri',
    'Anemia Failure to Grow Ulcerative Colitis and Weight',
    'Dynamic Flow A New Model for Ascorbate',
    'Integrative Medicine for Colon Cancer',
    'Screening for Vitamin C in the Urine Is it Clinically',
    'Orthomolecular and Botanical Treatments to Help',
    'Salvestrols–Natural Products with Tumour Selective',
    'Vitamin D and Health Implications for High-latitude',
    'Chronic Renal Disease Orthomolecular Ramifications',
    'Olfactory Sense and Supplements',
    'A School Phobia that Wasnt',
    'The Effects of a Low Glycemic Load Diet on Weight Loss',
    'Symptoms of Dissociation in a Forensic Population',
    'Reversing Splenomegalies in Epstein Barr Virus Infected',
    'Blood Basophils and Histamine Levels in Patients',
    'Heart Failure and Niacin',
    'Substance Abuse Trends Based on Drug-of-Choice Reports',
    'Special Report False Positive Finger Stick Blood',
    'The Successful Orthomolecular Treatment of AIDS',
    'Clinical Experiences with a Vitamin B3 Dependent Family',
    'Modeling Savings from Prophylactic REACT Antioxidant Use',
    'Poor Methodology in Meta-Analysis of Vitamins',
    'Schedule-Dependence in Cancer Therapy What is the True',
    'Safety and Effectiveness of Vitamins',
    'The Effect of High Dose IV Vitamin C on Plasma',
    'Vitamin K Deficiency Disease',
    'Vitamin C and the Common Cold',
    'The Real Story of Vitamin C and Cancer',
    'Vitamin C and Chemotherapy',
    'The Proper Treatment of Schizophrenia Requires Optimal',
    'Changes in Worker Fatigue after Vitamin C Administration',
    'Correspondence VE',
    'Antioxidant Vitamins Reduce the Risk for Cancer Part One',
    'Correspondence',
    'Vitamin D 25-OH-D3 Status of 200 Chronically Ill',
    'Antioxidant Vitamins Reduce the Risk for Cancer Part Two',
    'Role of Fat-Soluble Vitamins A and D in the Pathogenesis of Influenza',
    'Vitamin C and Hot Flashes'
]


querys_len = len(querys)
time_sum = 0
for i, query in enumerate(querys):
    start = time.time()
    # 数据库检索
    documents_str, references_str = vector_database.similarity_search(
        query = query,
    )
    end = time.time()
    time_sum += end - start
    logger.warning(f"{i+1}/{querys_len} search time: {end - start}")
    # logger.info(f"{i+1}/{querys_len} sdocuments_str: {documents_str}")

logger.error(f"average search time: {time_sum / querys_len}")
