select records.id, age, capital_gain, capital_loss, hours_week, over_50k, 
		countries.name as 'country', education_levels.name as 'education_level', 
		marital_statuses.name as 'marital_status', occupations.name as 'occupation',
		races.name as 'race', relationships.name as 'relationship_status', sexes.name as 'sex'
	from records join countries on
		records.country_id = countries.id
	join education_levels on
		records.education_level_id = education_levels.id
	join marital_statuses on
		records.marital_status_id = marital_statuses.id
	join occupations on
		records.occupation_id = occupations.id
	join races on
		records.race_id = races.id
	join relationships on
		records.relationship_id = relationships.id
	join sexes on
		records.sex_id = sexes.id
	join workclasses on
		records.workclass_id = workclasses.id;