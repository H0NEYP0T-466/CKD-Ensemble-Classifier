import React from 'react';
import type { FormFieldConfig } from '../types';

interface FormFieldProps {
  field: FormFieldConfig;
  value: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => void;
}

const FormField: React.FC<FormFieldProps> = ({ field, value, onChange }) => {
  const baseClasses = "form-group";

  if (field.type === 'select') {
    return (
      <div className={baseClasses}>
        <label htmlFor={field.name}>{field.label}</label>
        <select
          id={field.name}
          name={field.name}
          value={value}
          onChange={onChange}
          required={field.required}
        >
          {field.options?.map(option => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>
    );
  }

  return (
    <div className={baseClasses}>
      <label htmlFor={field.name}>{field.label}</label>
      <input
        type={field.type}
        id={field.name}
        name={field.name}
        value={value}
        onChange={onChange}
        required={field.required}
        min={field.min}
        max={field.max}
        step={field.step}
      />
    </div>
  );
};

export default FormField;